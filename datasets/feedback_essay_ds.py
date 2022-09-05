from torch.utils.data import Dataset, DataLoader
import torch
import collections
from transformers import AutoTokenizer, AutoConfig, AutoModel
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class CustomDataset(Dataset):
    def __init__(self, df, mode, cfg):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.architecture.backbone)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.tokenizer.sep_token is None:
            self.tokenizer.sep_token = " "

        if hasattr(cfg.dataset, "separator") and len(cfg.dataset.separator):
            self.cfg._tokenizer_sep_token = cfg.dataset.separator
        else:
            self.cfg._tokenizer_sep_token = self.tokenizer.sep_token

        self.text = self.get_texts(self.df, self.cfg, self.tokenizer.sep_token)

        self.label_cols = cfg.dataset.label_columns

        if self.label_cols == ["discourse_type"]:
            self.labels = (
                OneHotEncoder()
                .fit_transform(self.df.discourse_type.values.reshape(-1, 1))
                .A
            )
        else:
            self.labels = self.df[self.label_cols].values

        self.weights = self.labels.max(axis=1)

        if self.cfg.dataset.group_discourse:
            self.df["weights"] = self.weights
            self.df.loc[
                self.df.discourse_id == "56744a66949a", "discourse_text"
            ] = "This whole thing is point less how they have us in here for two days im missing my education. We could have finished this in one day and had the rest of the week to get back on the track of learning. I've missed both days of weight lifting, algebra, and my world history that i do not want to fail again! If their are any people actually gonna sit down and take the time to read this then\n\nDO NOT DO THIS NEXT YEAR\n\n.\n\nThey are giving us cold lunches. ham and cheese and an apple, I am 16 years old and my body needs proper food. I wouldnt be complaining if they served actual breakfast. but because of Michelle Obama and her healthy diet rule they surve us 1 poptart in the moring. How does the school board expect us to last from 7:05-12:15 on a pop tart? then expect us to get A's, we are more focused on lunch than anything else. I am about done so if you have the time to read this even though this does not count. Bring PROPER_NAME a big Mac from mc donalds, SCHOOL_NAME, (idk area code but its in LOCATION_NAME)       \xa0    "

            grps = self.df.groupby("essay_id", sort=False)
            self.grp_texts = []
            self.grp_labels = []
            self.grp_weights = []

            s = 0

            if self.mode == "train" and self.cfg.architecture.aux_type:
                self.grp_labels_aux = []
                self.df["aux_labels"] = [
                    torch.Tensor(np.array(x))
                    for x in OneHotEncoder()
                    .fit_transform(self.df.discourse_type.values.reshape(-1, 1))
                    .A
                ]

            for jjj, grp in enumerate(grps.groups):
                g = grps.get_group(grp)
                t = g.essay_text.values[0]

                labels = []
                labels_aux = []
                weights = []

                end = 0
                for j in range(len(g)):

                    labels.append(g[self.label_cols].values[j])
                    weights.append(g["weights"].values[j])

                    if self.mode == "train" and self.cfg.architecture.aux_type:
                        labels_aux.append(g["aux_labels"].values[j])

                    d = g.discourse_text.values[j]

                    start = t[end:].find(d.strip())
                    if start == -1:
                        print("ERROR")

                    start = start + end

                    end = start + len(d.strip())

                    if (
                        hasattr(self.cfg.architecture, "use_sep")
                        and self.cfg.architecture.use_sep
                    ):
                        t = (
                            t[:start]
                            + f" {self.cfg._tokenizer_sep_token} "
                            + t[start:end]
                            + f" {self.cfg._tokenizer_sep_token} "
                            + t[end:]
                        )

                    elif self.cfg.architecture.aux_type:
                        t = (
                            t[:start]
                            + f" [START] "
                            + t[start:end]
                            + " [END] "
                            + t[end:]
                        )
                    elif self.cfg.architecture.use_type:
                        t = (
                            t[:start]
                            + f" [START_{g.discourse_type.values[j]}]  "
                            + t[start:end]
                            + f" [END_{g.discourse_type.values[j]}] "
                            + t[end:]
                        )
                    elif (
                        hasattr(self.cfg.architecture, "no_type")
                        and self.cfg.architecture.no_type
                    ):
                        t = (
                            t[:start]
                            + f" [START]  "
                            + t[start:end]
                            + " [END] "
                            + t[end:]
                        )
                    else:
                        t = (
                            t[:start]
                            + f" [START] {g.discourse_type.values[j]} "
                            + t[start:end]
                            + " [END] "
                            + t[end:]
                        )

                if self.cfg.dataset.add_group_types:
                    t = (
                        " ".join(g.discourse_type.values)
                        + f" {self.cfg._tokenizer_sep_token} "
                        + t
                    )

                self.grp_texts.append(t)
                self.grp_labels.append(labels)
                self.grp_weights.append(weights)
                if self.mode == "train" and self.cfg.architecture.aux_type:
                    self.grp_labels_aux.append(torch.stack(labels_aux))
                s += len(g)

                if jjj == 0:
                    print(t)
                    print(labels)

            if self.cfg.dataset.group_discourse:

                self.cfg._tokenizer_start_token_id = []
                self.cfg._tokenizer_end_token_id = []

                if self.cfg.architecture.use_type:
                    for type in sorted(self.df.discourse_type.unique()):
                        self.tokenizer.add_tokens(
                            [f"[START_{type}]"], special_tokens=True
                        )
                        self.cfg._tokenizer_start_token_id.append(
                            self.tokenizer.encode(f"[START_{type}]")[1]
                        )

                    for type in sorted(self.df.discourse_type.unique()):
                        self.tokenizer.add_tokens(
                            [f"[END_{type}]"], special_tokens=True
                        )
                        self.cfg._tokenizer_end_token_id.append(
                            self.tokenizer.encode(f"[END_{type}]")[1]
                        )

                else:
                    self.tokenizer.add_tokens(["[START]", "[END]"], special_tokens=True)
                    idx = 1
                    self.cfg._tokenizer_start_token_id.append(
                        self.tokenizer.encode(f"[START]")[idx]
                    )
                    self.cfg._tokenizer_end_token_id.append(
                        self.tokenizer.encode(f"[END]]")[idx]
                    )

                print(self.cfg._tokenizer_start_token_id)
                print(self.cfg._tokenizer_end_token_id)

            if (
                hasattr(self.cfg.tokenizer, "add_newline_token")
                and self.cfg.tokenizer.add_newline_token
            ):
                self.tokenizer.add_tokens([f"\n"], special_tokens=True)

            self.cfg._tokenizer_size = len(self.tokenizer)
        else:
            print(self.text[0])

        self.cfg._tokenizer_cls_token_id = self.tokenizer.cls_token_id
        self.cfg._tokenizer_sep_token_id = self.tokenizer.sep_token_id
        self.cfg._tokenizer_mask_token_id = self.tokenizer.mask_token_id

    def __len__(self):
        if self.cfg.dataset.group_discourse:
            return len(self.grp_texts)
        else:
            return len(self.df)

    @staticmethod
    def collate_fn(batch):
        elem = batch[0]

        ret = {}
        for key in elem:
            if key in {"target", "weight"}:
                ret[key] = [d[key].float() for d in batch]
            elif key in {"target_aux"}:

                ret[key] = [d[key].float() for d in batch]
            else:
                ret[key] = torch.stack([d[key] for d in batch], 0)
        return ret

    @staticmethod
    def get_train_collate_fn(cfg):
        if cfg.dataset.group_discourse:
            return CustomDataset.collate_fn
        else:
            return None

    @staticmethod
    def get_validation_collate_fn(cfg):
        if cfg.dataset.group_discourse:
            return CustomDataset.collate_fn
        else:
            return None

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

    def get_texts(cls, df, cfg, separator):
        if separator is None:
            if hasattr(cfg.dataset, "separator") and len(cfg.dataset.separator):
                separator = cfg.dataset.separator
            else:
                separator = getattr(cfg, "_tokenizer_sep_token", "<SEPARATOR>")

        if isinstance(cfg.dataset.text_column, str):
            texts = df[cfg.dataset.text_column].astype(str)

            texts = texts.values
        else:
            columns = list(cfg.dataset.text_column)
            join_str = f" {separator} "
            texts = df[columns].astype(str)

            texts = texts.apply(lambda x: join_str.join(x), axis=1).values

        return texts

    def _read_data(self, idx, sample):
        if self.cfg.dataset.group_discourse:
            text = self.grp_texts[idx]
        else:
            text = self.text[idx]

        if idx == 0:
            print(text)

        sample.update(self.encode(text))
        return sample

    def encode(self, text):
        sample = dict()
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.tokenizer.max_length,
        )
        sample["input_ids"] = encodings["input_ids"][0]
        sample["attention_mask"] = encodings["attention_mask"][0]
        return sample

    def _read_label(self, idx, sample):
        if (
            hasattr(self.cfg.dataset, "group_discourse")
            and self.cfg.dataset.group_discourse
        ):
            sample["target"] = self.grp_labels[idx]
            if self.mode == "train" and self.cfg.architecture.aux_type:
                sample["target_aux"] = self.grp_labels_aux[idx]
        else:
            sample["target"] = self.labels[idx]
        return sample

    def __getitem__(self, idx):
        sample = dict()

        sample = self._read_data(idx=idx, sample=sample)

        if self.label_cols is not None:
            sample = self._read_label(idx=idx, sample=sample)

        if "target" in sample:
            sample["target"] = torch.tensor(np.array(sample["target"])).float()

        return sample
