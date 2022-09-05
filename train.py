import numpy as np
import pandas as pd
import importlib
import sys
import random
from tqdm import tqdm
import gc
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import log_loss
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import SequentialSampler, DataLoader
import yaml
from types import SimpleNamespace
import os
import re


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append("models")
sys.path.append("datasets")


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_train_dataloader(train_ds, cfg):
    train_dataloader = DataLoader(
        train_ds,
        sampler=None,
        shuffle=True,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.environment.number_of_workers,
        pin_memory=False,
        collate_fn=cfg.CustomDataset.get_train_collate_fn(cfg),
        drop_last=cfg.training.drop_last_batch,
        worker_init_fn=worker_init_fn,
    )
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_val_dataloader(val_ds, cfg):
    val_dataloader = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=1,
        num_workers=cfg.environment.number_of_workers,
        pin_memory=True,
        collate_fn=cfg.CustomDataset.get_validation_collate_fn(cfg),
        worker_init_fn=worker_init_fn,
    )
    print(f"val: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader


def get_scheduler(cfg, optimizer, total_steps):
    if cfg.training.schedule == "Linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                cfg.training.warmup_epochs * (total_steps // cfg.training.batch_size)
            ),
            num_training_steps=cfg.training.epochs
            * (total_steps // cfg.training.batch_size),
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                cfg.training.warmup_epochs * (total_steps // cfg.training.batch_size)
            ),
            num_training_steps=cfg.training.epochs
            * (total_steps // cfg.training.batch_size),
        )
    return scheduler


def load_checkpoint(cfg, model):
    d = torch.load(cfg.architecture.pretrained_weights, map_location="cpu")

    if "model" in d:
        model_weights = d["model"]
    else:
        model_weights = d

    if (
        model.backbone.embeddings.word_embeddings.weight.shape[0]
        < model_weights["backbone.embeddings.word_embeddings.weight"].shape[0]
    ):
        print("resizing pretrained embedding weights")
        model_weights["backbone.embeddings.word_embeddings.weight"] = model_weights[
            "backbone.embeddings.word_embeddings.weight"
        ][: model.backbone.embeddings.word_embeddings.weight.shape[0]]

    try:
        model.load_state_dict(model_weights, strict=True)
    except Exception as e:
        print("removing unused pretrained layers")
        for layer_name in re.findall("size mismatch for (.*?):", str(e)):
            model_weights.pop(layer_name, None)
        model.load_state_dict(model_weights, strict=False)

    print(f"Weights loaded from: {cfg.architecture.pretrained_weights}")


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_model(cfg):
    Net = importlib.import_module(cfg.model_class).Net
    return Net(cfg)


parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)
cfg = yaml.safe_load(open(parser_args.config).read())
for k, v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**cfg)
print(cfg)

os.makedirs(f"output/{cfg.experiment_name}", exist_ok=True)
cfg.CustomDataset = importlib.import_module(cfg.dataset_class).CustomDataset

if __name__ == "__main__":
    device = "cuda:0"
    cfg.device = device

    if cfg.environment.seed < 0:
        cfg.environment.seed = np.random.randint(1_000_000)
    else:
        cfg.environment.seed = cfg.environment.seed

    set_seed(cfg.environment.seed)

    if cfg.dataset.train_dataframe.endswith(".pq"):
        train_df = pd.read_parquet(cfg.dataset.train_dataframe)
    else:
        train_df = pd.read_csv(cfg.dataset.train_dataframe)

    if "fold" in train_df.columns:
        if cfg.dataset.fold == -1:
            val_df = train_df[train_df.fold == 0].copy()
        else:
            val_df = train_df[train_df.fold == cfg.dataset.fold].copy()
            train_df = train_df[train_df.fold != cfg.dataset.fold].copy()
    else:
        val_df = train_df.copy()

    train_dataset = cfg.CustomDataset(train_df, mode="train", cfg=cfg)
    val_dataset = cfg.CustomDataset(val_df, mode="val", cfg=cfg)

    cfg.train_dataset = train_dataset

    train_dataloader = get_train_dataloader(train_dataset, cfg)
    val_dataloader = get_val_dataloader(val_dataset, cfg)

    model = get_model(cfg)
    model.to(device)

    if cfg.architecture.pretrained_weights != "":
        try:
            load_checkpoint(cfg, model)
        except:
            print("WARNING: could not load checkpoint")

    total_steps = len(train_dataset)

    params = model.parameters()

    no_decay = ["bias", "LayerNorm.weight"]
    differential_layers = cfg.training.differential_learning_rate_layers
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.learning_rate,
                "weight_decay": cfg.training.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.learning_rate,
                "weight_decay": 0,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.differential_learning_rate,
                "weight_decay": cfg.training.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.differential_learning_rate,
                "weight_decay": 0,
            },
        ],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = get_scheduler(cfg, optimizer, total_steps)

    if cfg.environment.mixed_precision:
        scaler = GradScaler()

    cfg.curr_step = 0
    i = 0
    best_val_loss = np.inf
    optimizer.zero_grad()
    for epoch in range(cfg.training.epochs):
        set_seed(cfg.environment.seed + epoch)

        cfg.curr_epoch = epoch
        print("EPOCH:", epoch)

        progress_bar = tqdm(range(len(train_dataloader)))
        tr_it = iter(train_dataloader)
        losses = []
        gc.collect()
        model.train()

        # ==== TRAIN LOOP
        for itr in progress_bar:
            i += 1
            cfg.curr_step += cfg.training.batch_size
            data = next(tr_it)
            batch = cfg.CustomDataset.batch_to_device(data, device)

            if cfg.environment.mixed_precision:
                with autocast():
                    output_dict = model(batch)
            else:
                output_dict = model(batch)

            loss = output_dict["loss"]

            losses.append(loss.item())

            if cfg.training.grad_accumulation != 1:
                loss /= cfg.training.grad_accumulation

            if cfg.environment.mixed_precision:
                scaler.scale(loss).backward()
                if i % cfg.training.grad_accumulation == 0:
                    if cfg.training.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.training.gradient_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if i % cfg.training.grad_accumulation == 0:
                    if cfg.training.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.training.gradient_clip
                        )
                    optimizer.step()
                    optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            if cfg.curr_step % cfg.training.batch_size == 0:
                progress_bar.set_description(
                    f"lr: {np.round(optimizer.param_groups[0]['lr'],7)}, loss: {np.mean(losses[-10:]):.4f}"
                )

        progress_bar = tqdm(range(len(val_dataloader)))
        val_it = iter(val_dataloader)

        model.eval()
        preds = []
        probabilities = []
        all_targets = []
        for itr in progress_bar:
            data = next(val_it)
            batch = cfg.CustomDataset.batch_to_device(data, device)

            if cfg.environment.mixed_precision:
                with autocast():
                    output_dict = model(batch)
            else:
                output_dict = model(batch)

            if cfg.dataset_class == "feedback_dataset":
                for logits, word_start_mask, target in zip(
                    output_dict["logits"],
                    output_dict["word_start_mask"],
                    output_dict["target"],
                ):
                    probs = (
                        torch.softmax(logits[word_start_mask], dim=1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    targets = target[word_start_mask].detach().cpu().numpy()

                    for p, t in zip(probs, targets):
                        probabilities.append(p)
                        if cfg.training.is_pseudo:
                            all_targets.append(np.argmax(t))
                        else:
                            all_targets.append(t)
            else:
                preds.append(
                    output_dict["logits"].float().softmax(dim=1).detach().cpu().numpy()
                )

        if cfg.dataset_class == "feedback_dataset":
            metric = log_loss(
                all_targets,
                np.vstack(probabilities),
                eps=1e-7,
                labels=list(range(cfg.dataset.num_classes)),
            )
        else:
            preds = np.concatenate(preds, axis=0)
            if cfg.experiment_name != "pretrain-type-v9":
                metric = log_loss(
                    val_df[cfg.dataset.label_columns].values.argmax(axis=1), preds
                )
            else:
                metric = None
        print("Validation metric", metric)

        if cfg.training.epochs > 0:
            checkpoint = {"model": model.state_dict()}
            torch.save(checkpoint, f"output/{cfg.experiment_name}/checkpoint.pth")
