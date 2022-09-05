import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from torch.nn.parameter import Parameter


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target, weights=None):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class WeightedDenseCrossEntropy(nn.Module):
    def forward(self, x, target, weights=None):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)

        if weights is not None:
            loss = loss * weights
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.mean()

        return loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor, weights=None):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor.argmax(dim=1),
            weight=self.weight,
            reduction=self.reduction,
        )


class NLPAllclsTokenPooling(nn.Module):
    def __init__(self, dim, cfg):
        super(NLPAllclsTokenPooling, self).__init__()

        self.dim = dim
        self.feat_mult = 1
        if cfg.dataset.group_discourse:
            self.feat_mult = 3

    def forward(self, x, attention_mask, input_ids, cfg):

        if not cfg.dataset.group_discourse:
            input_ids_expanded = input_ids.clone().unsqueeze(-1).expand(x.shape)
            attention_mask_expanded = torch.zeros_like(input_ids_expanded)

            attention_mask_expanded[
                (input_ids_expanded == cfg._tokenizer_cls_token_id)
                | (input_ids_expanded == cfg._tokenizer_sep_token_id)
            ] = 1

            sum_features = (x * attention_mask_expanded).sum(self.dim)
            ret = sum_features / attention_mask_expanded.sum(self.dim).clip(min=1e-8)

        else:
            ret = []

            for j in range(x.shape[0]):

                idx0 = torch.where(
                    (input_ids[j] >= min(cfg._tokenizer_start_token_id))
                    & (input_ids[j] <= max(cfg._tokenizer_start_token_id))
                )[0]
                idx1 = torch.where(
                    (input_ids[j] >= min(cfg._tokenizer_end_token_id))
                    & (input_ids[j] <= max(cfg._tokenizer_end_token_id))
                )[0]

                xx = []
                for jj in range(len(idx0)):
                    xx0 = x[j, idx0[jj]]
                    xx1 = x[j, idx1[jj]]
                    xx2 = x[j, idx0[jj] + 1 : idx1[jj]].mean(dim=0)
                    xxx = torch.cat([xx0, xx1, xx2]).unsqueeze(0)
                    xx.append(xxx)
                xx = torch.cat(xx)
                ret.append(xx)

        return ret


class GeMText(nn.Module):
    def __init__(self, dim, cfg, p=3, eps=1e-6):
        super(GeMText, self).__init__()
        self.dim = dim
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.feat_mult = 1

    def forward(self, x, attention_mask, input_ids, cfg):
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(x.shape)
        x = (x.clamp(min=self.eps) * attention_mask_expanded).pow(self.p).sum(self.dim)
        ret = x / attention_mask_expanded.sum(self.dim).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret


class NLPPoolings:
    _poolings = {"All [CLS] token": NLPAllclsTokenPooling, "GeM": GeMText}

    @classmethod
    def get(cls, name):
        return cls._poolings.get(name)


class Net(nn.Module):
    def __init__(self, cfg):

        super(Net, self).__init__()

        self.cfg = cfg

        config = AutoConfig.from_pretrained(cfg.architecture.backbone)
        if cfg.architecture.custom_intermediate_dropout:
            config.hidden_dropout_prob = cfg.architecture.intermediate_dropout
            config.attention_probs_dropout_prob = cfg.architecture.intermediate_dropout
        self.backbone = AutoModel.from_pretrained(
            cfg.architecture.backbone, config=config
        )

        if self.cfg.dataset.group_discourse:
            self.backbone.resize_token_embeddings(cfg._tokenizer_size)
        print("Embedding size", self.backbone.embeddings.word_embeddings.weight.shape)

        self.pooling = NLPPoolings.get(self.cfg.architecture.pool)
        self.pooling = self.pooling(dim=1, cfg=cfg)

        dim = self.backbone.config.hidden_size * self.pooling.feat_mult

        self.dim = dim

        if cfg.dataset.label_columns == ["discourse_type"]:
            self.head = nn.Linear(dim, 7)
        else:
            self.head = nn.Linear(dim, 3)

        if self.cfg.architecture.aux_type:
            self.aux_classifier = nn.Linear(dim, 7)

        if self.cfg.training.loss_function == "CrossEntropy":
            self.loss_fn = DenseCrossEntropy()
        elif self.cfg.training.loss_function == "WeightedCrossEntropy":
            self.loss_fn = WeightedDenseCrossEntropy()
        elif self.cfg.training.loss_function == "FocalLoss":
            self.loss_fn = FocalLoss()

        if self.cfg.training.aux_loss_function == "CrossEntropy":
            self.loss_fn_aux = DenseCrossEntropy()
        elif self.cfg.training.aux_loss_function == "FocalLoss":
            self.loss_fn_aux = FocalLoss()

    def _num_outputs(self):
        return self.backbone.config.hidden_size * self.pooling.feat_mult

    def get_features(self, batch):
        attention_mask = batch["attention_mask"]
        input_ids = batch["input_ids"]

        x = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        x = self.pooling(x, attention_mask, input_ids, self.cfg)

        if self.cfg.dataset.group_discourse:
            batch["target"] = [batch["target"][j][: len(x[j])] for j in range(len(x))]
            batch["target"] = torch.cat(batch["target"])

            if self.training and self.cfg.architecture.aux_type:
                batch["target_aux"] = [
                    batch["target_aux"][j][: len(x[j])] for j in range(len(x))
                ]
                batch["target_aux"] = torch.cat(batch["target_aux"])

            x = torch.cat(x)

        return x, batch

    def forward(self, batch, calculate_loss=True):
        idx = int(torch.where(batch["attention_mask"] == 1)[1].max())
        idx += 1
        batch["attention_mask"] = batch["attention_mask"][:, :idx]
        batch["input_ids"] = batch["input_ids"][:, :idx]

        if (
            self.training
            and hasattr(self.cfg.training, "mask_probability")
            and self.cfg.training.mask_probability > 0
        ):
            input_ids = batch["input_ids"].clone()
            special_mask = torch.ones_like(input_ids)
            special_mask[
                (input_ids == self.cfg._tokenizer_cls_token_id)
                | (input_ids == self.cfg._tokenizer_sep_token_id)
                | (input_ids >= self.cfg._tokenizer_mask_token_id)
            ] = 0
            mask = (
                torch.bernoulli(
                    torch.full(input_ids.shape, self.cfg.training.mask_probability)
                )
                .to(input_ids.device)
                .bool()
                & special_mask
            ).bool()
            input_ids[mask] = self.cfg._tokenizer_mask_token_id
            batch["input_ids"] = input_ids.clone()

        x, batch = self.get_features(batch)
        if self.cfg.architecture.dropout > 0.0:
            x = F.dropout(x, p=self.cfg.architecture.dropout, training=self.training)

        if (
            hasattr(self.cfg.architecture, "wide_dropout")
            and self.cfg.architecture.wide_dropout > 0.0
            and self.training
        ):
            x1 = self.head(
                F.dropout(
                    x, p=self.cfg.architecture.wide_dropout, training=self.training
                )
            )
            x2 = self.head(
                F.dropout(
                    x, p=self.cfg.architecture.wide_dropout, training=self.training
                )
            )
            x3 = self.head(
                F.dropout(
                    x, p=self.cfg.architecture.wide_dropout, training=self.training
                )
            )
            x4 = self.head(
                F.dropout(
                    x, p=self.cfg.architecture.wide_dropout, training=self.training
                )
            )
            x5 = self.head(
                F.dropout(
                    x, p=self.cfg.architecture.wide_dropout, training=self.training
                )
            )
            logits = (x1 + x2 + x3 + x4 + x5) / 5
        else:
            logits = self.head(x)

        if self.cfg.architecture.aux_type:
            logits_aux = self.aux_classifier(x)

        outputs = {}

        outputs["logits"] = logits

        if "target" in batch:
            outputs["target"] = batch["target"]

        if calculate_loss:
            targets = batch["target"]

            if self.training:
                outputs["loss"] = self.loss_fn(logits, targets)

                if self.cfg.architecture.aux_type:
                    outputs["loss_reg"] = outputs["loss"]
                    outputs["loss_aux"] = self.loss_fn_aux(
                        logits_aux, batch["target_aux"]
                    )
                    outputs["loss"] = (
                        0.5 * outputs["loss_reg"] + 0.5 * outputs["loss_aux"]
                    )

        return outputs
