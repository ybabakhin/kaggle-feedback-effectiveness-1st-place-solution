import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from torch.utils.checkpoint import checkpoint


def batch_padding(cfg, batch):
    idx = int(torch.where(batch["attention_mask"] == 1)[1].max())

    idx += 1
    batch["attention_mask"] = batch["attention_mask"][:, :idx]
    batch["input_ids"] = batch["input_ids"][:, :idx]
    batch["target"] = batch["target"][:, :idx]
    batch["word_start_mask"] = batch["word_start_mask"][:, :idx]
    batch["word_ids"] = batch["word_ids"][:, :idx]

    return batch


# https://www.kaggle.com/code/cpmpml/sub-ensemble-010/notebook?scriptVersionId=94177123
def glorot_uniform(parameter):
    nn.init.xavier_uniform_(parameter.data, gain=1.0)


class NBMEHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NBMEHead, self).__init__()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.classifier = nn.Linear(input_dim, output_dim)
        glorot_uniform(self.classifier.weight)

    def forward(self, x):
        # x is B x S x C
        logits1 = self.classifier(self.dropout1(x))
        logits2 = self.classifier(self.dropout2(x))
        logits3 = self.classifier(self.dropout3(x))
        logits4 = self.classifier(self.dropout4(x))
        logits5 = self.classifier(self.dropout5(x))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        return logits


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()

        self.cfg = cfg

        config = AutoConfig.from_pretrained(cfg.architecture.backbone)
        self.backbone = AutoModel.from_pretrained(
            cfg.architecture.backbone, config=config
        )

        if cfg.architecture.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self.head = nn.Linear(self.backbone.config.hidden_size, cfg.dataset.num_classes)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        if (
            hasattr(self.cfg.architecture, "add_wide_dropout")
            and self.cfg.architecture.add_wide_dropout
        ):
            self.token_type_head = NBMEHead(
                self.backbone.config.hidden_size, cfg.dataset.num_classes
            )

    def forward(self, batch, calculate_loss=True):
        outputs = {}

        if calculate_loss:
            outputs["target"] = batch["target"]
            outputs["word_start_mask"] = batch["word_start_mask"]

        batch = batch_padding(self.cfg, batch)

        x = self.backbone(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state

        for obs_id in range(x.size()[0]):
            for w_id in range(int(torch.max(batch["word_ids"][obs_id]).item()) + 1):
                chunk_mask = batch["word_ids"][obs_id] == w_id
                chunk_logits = x[obs_id] * chunk_mask.unsqueeze(-1)
                chunk_logits = chunk_logits.sum(dim=0) / chunk_mask.sum()
                x[obs_id][chunk_mask] = chunk_logits

        if (
            hasattr(self.cfg.architecture, "add_wide_dropout")
            and self.cfg.architecture.add_wide_dropout
        ):
            logits = self.token_type_head(x)
        else:
            if self.cfg.architecture.dropout > 0.0:
                x = F.dropout(
                    x, p=self.cfg.architecture.dropout, training=self.training
                )

            logits = self.head(x)

        if not self.training:
            outputs["logits"] = F.pad(
                logits,
                (0, 0, 0, self.cfg.tokenizer.max_length - logits.size()[1]),
                "constant",
                0,
            )

        if calculate_loss:
            targets = batch["target"]
            new_logits = logits.view(-1, self.cfg.dataset.num_classes)

            if self.cfg.training.is_pseudo:
                new_targets = targets.reshape(-1, self.cfg.dataset.num_classes)
            else:
                new_targets = targets.reshape(-1)
            new_word_start_mask = batch["word_start_mask"].reshape(-1)

            loss = self.loss_fn(new_logits, new_targets)
            outputs["loss"] = (
                loss * new_word_start_mask
            ).sum() / new_word_start_mask.sum()

        return outputs
