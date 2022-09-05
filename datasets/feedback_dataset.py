import collections

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoConfig, AutoModel


class CustomDataset(Dataset):
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = dict()
        sample = self._read_data(idx=idx, sample=sample)
        sample = self._read_label(idx=idx, sample=sample)

        return sample

    def __init__(self, df, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()

        if mode == "train" and not cfg.training.is_pseudo:
            cfg._label_encoder = LabelEncoder().fit(
                np.concatenate(self.df[cfg.dataset.label_columns].values)
            )

        if not cfg.training.is_pseudo:
            self.df[cfg.dataset.label_columns] = self.df[cfg.dataset.label_columns].map(
                lambda labels: cfg._label_encoder.transform(labels)
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.architecture.backbone, add_prefix_space=True, use_fast=True
        )

        self.text = self.df[cfg.dataset.text_column].values
        self.labels = self.df[self.cfg.dataset.label_columns].values

    def _read_data(self, idx, sample):
        text = self.text[idx]

        if hasattr(self.cfg.training, "add_types") and self.cfg.training.add_types:
            tokenizer_input = [
                x if x_idx > 0 else x + self.tokenizer.sep_token
                for x_idx, x in enumerate(list(text))
            ]
        else:
            tokenizer_input = [list(text)]

        encodings = self.tokenizer(
            tokenizer_input,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.tokenizer.max_length,
            is_split_into_words=True,
        )

        sample["input_ids"] = encodings["input_ids"][0]
        sample["attention_mask"] = encodings["attention_mask"][0]

        word_ids = encodings.word_ids(0)
        word_ids = [-1 if x is None else x for x in word_ids]
        sample["word_ids"] = torch.tensor(word_ids)

        word_start_mask = []
        lab_idx = -1

        for i, word in enumerate(word_ids):
            word_start = word > -1 and (i == 0 or word_ids[i - 1] != word)
            if word_start:
                lab_idx += 1
                if self.cfg.training.is_pseudo:
                    if self.labels[idx][lab_idx][0] > 0:
                        word_start_mask.append(True)
                        continue
                else:
                    if self.labels[idx][lab_idx] != self.cfg.dataset.num_classes:
                        word_start_mask.append(True)
                        continue

            word_start_mask.append(False)

        sample["word_start_mask"] = torch.tensor(word_start_mask)

        return sample

    def _read_label(self, idx, sample):
        if self.cfg.training.is_pseudo:
            target = torch.ones(2048, self.cfg.dataset.num_classes) - 101
        else:
            target = torch.full_like(sample["input_ids"], -100)

        word_start_mask = sample["word_start_mask"]

        if self.cfg.training.is_pseudo:
            target[word_start_mask] = torch.tensor(
                [x.astype(np.float32) for x in self.labels[idx] if float(x[0]) > -0.01]
            )
        else:
            target[word_start_mask] = torch.tensor(
                [x for x in self.labels[idx] if x != self.cfg.dataset.num_classes]
            )

        sample["target"] = target
        return sample

    @staticmethod
    def get_train_collate_fn(cfg):
        return None

    @staticmethod
    def get_validation_collate_fn(cfg):
        return None

    @staticmethod
    def batch_to_device(batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, collections.abc.Mapping):
            return {
                key: CustomDataset.batch_to_device(value, device)
                for key, value in batch.items()
            }
        elif isinstance(batch, collections.abc.Sequence):
            return [CustomDataset.batch_to_device(value, device) for value in batch]
        else:
            raise ValueError(f"Can not move {type(batch)} to device.")
