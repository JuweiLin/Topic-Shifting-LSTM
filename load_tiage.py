# load_tiage.py

import json
import torch
from torch.utils.data import Dataset, DataLoader


class TIAGEDataset(Dataset):
    def __init__(self, json_path, tokenizer, sent_token="<sent>"):
        self.data = []
        self.tokenizer = tokenizer
        self.sent_token = sent_token
        self._load(json_path)

    def _load(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        self.data = obj["dial_data"]["tiage"]

    def __len__(self):
        return len(self.data)

    def _map_role(self, role_str):
        if role_str is None:
            return 0
        r = str(role_str).lower()
        if r == "agent":
            return 0
        if r == "user":
            return 1
        return 0

    def _map_label(self, raw_label):
        if raw_label is None:
            return 0
        try:
            v = int(raw_label)
        except Exception:
            return 0
        return 1 if v != 0 else 0

    def __getitem__(self, idx):
        dialog = self.data[idx]

        dialog_id = dialog.get("dial_id", str(idx))
        turns = dialog.get("turns", [])

        all_ids = []
        sent_spans = []
        labels = []
        roles = []
        sent_len = []
        position = []

        for i, turn in enumerate(turns):
            text = turn.get("utterance", "")
            raw_label = turn.get("segmentation_label", 0)
            raw_role = turn.get("role", "")

            label = self._map_label(raw_label)
            role_id = self._map_role(raw_role)

            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False
            )

            start = len(all_ids)
            all_ids.extend(encoded)
            end = len(all_ids)

            if end == start:
                pad_id = self.tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = 0
                all_ids.append(pad_id)
                start = len(all_ids) - 1
                end = len(all_ids)

            sent_spans.append((start, end))
            labels.append(label)
            roles.append(role_id)
            sent_len.append(end - start)
            position.append(i)

        input_ids = torch.tensor(all_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        labels = torch.tensor(labels, dtype=torch.long)
        roles = torch.tensor(roles, dtype=torch.long)
        sent_len = torch.tensor(sent_len, dtype=torch.long)
        position = torch.tensor(position, dtype=torch.long)

        return {
            "dialog_id": dialog_id,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sent_spans": sent_spans,
            "labels": labels,
            "roles": roles,
            "sent_len": sent_len,
            "position": position,
        }


def collate_batch(batch):

    B = len(batch)

    lengths = [item["input_ids"].size(0) for item in batch]
    L_max = max(lengths)

    input_ids = torch.zeros(B, L_max, dtype=torch.long)
    attention_mask = torch.zeros(B, L_max, dtype=torch.long)

    for i, item in enumerate(batch):
        l = item["input_ids"].size(0)
        input_ids[i, :l] = item["input_ids"]
        attention_mask[i, :l] = item["attention_mask"]

    Ks = [item["labels"].size(0) for item in batch]
    K_max = max(Ks)

    labels = torch.zeros(B, K_max, dtype=torch.long)
    roles = torch.zeros(B, K_max, dtype=torch.long)
    sent_len = torch.zeros(B, K_max, dtype=torch.long)
    position = torch.zeros(B, K_max, dtype=torch.long)
    sent_mask = torch.zeros(B, K_max, dtype=torch.float)

    sent_spans_batch = []

    for i, item in enumerate(batch):
        K = item["labels"].size(0)
        labels[i, :K] = item["labels"]
        roles[i, :K] = item["roles"]
        sent_len[i, :K] = item["sent_len"]
        position[i, :K] = item["position"]
        sent_mask[i, :K] = 1.0

        sent_spans_batch.append(item["sent_spans"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "sent_spans": sent_spans_batch,
        "sent_mask": sent_mask,
        "labels": labels,
        "roles": roles,
        "sent_len": sent_len,
        "position": position,
    }

def build_dataloader(json_path, tokenizer, batch_size_dialog, shuffle=False, num_workers=0):
    dataset = TIAGEDataset(json_path, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size_dialog,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
    return loader
