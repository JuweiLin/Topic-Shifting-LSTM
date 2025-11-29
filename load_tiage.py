# load_tiage.py

import json
import torch
from torch.utils.data import Dataset, DataLoader
from tables import ROLE2ID, DISCOURSE_MARKERS


class TIAGEDataset(Dataset):
    def __init__(self, json_path, tokenizer, sent_token="<sent>"):
        self.data = []
        self.tokenizer = tokenizer
        self.sent_token = sent_token
        self._load(json_path)

    def _load(self, json_path):
        # 统一成 list[路径]
        if isinstance(json_path, (list, tuple)):
            paths = list(json_path)
        else:
            paths = [json_path]

        dialogs_all = []

        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            # 适配 {"dial_data": {"tiage": [ ... ]}} 这种结构
            if isinstance(raw, dict) and "dial_data" in raw and "tiage" in raw["dial_data"]:
                dialogs = raw["dial_data"]["tiage"]
            else:
                dialogs = raw

            dialogs_all.extend(dialogs)

        # 存成 list[dialog]
        self.data = dialogs_all


    def __len__(self):
        return len(self.data)

    def _map_role(self, raw_role: str) -> int:
        key = (raw_role or "").lower()
        return ROLE2ID.get(key, ROLE2ID["other"])

    def _map_label(self, raw_label) -> int:
        return int(raw_label)

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

        # 新增：句级 feature 的原始值
        is_question = []
        disc_marker = []

        for i, turn in enumerate(turns):
            text = turn.get("utterance", "")
            raw_label = turn.get("segmentation_label", 0)
            raw_role = turn.get("role", "")

            label = self._map_label(raw_label)   # 0/1
            role_id = self._map_role(raw_role)   # user/agent -> 0/1/2

            # is_question：当前句是否问句
            txt_strip = text.strip()
            is_q = 1 if txt_strip.endswith("?") else 0
            is_question.append(is_q)

            # disc_marker：是否包含显式话题标记
            lower = text.lower()
            has_marker = 1 if any(m in lower for m in DISCOURSE_MARKERS) else 0
            disc_marker.append(has_marker)

            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False
            )

            start = len(all_ids)
            all_ids.extend(encoded)
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
        is_question = torch.tensor(is_question, dtype=torch.long)
        disc_marker = torch.tensor(disc_marker, dtype=torch.long)

        return {
            "dialog_id": dialog_id,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sent_spans": sent_spans,
            "labels": labels,
            "roles": roles,
            "sent_len": sent_len,
            "position": position,
            "is_question": is_question,
            "disc_marker": disc_marker,
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

    # 新增：问句 + marker
    is_question = torch.zeros(B, K_max, dtype=torch.long)
    disc_marker = torch.zeros(B, K_max, dtype=torch.long)

    sent_spans_batch = []

    for i, item in enumerate(batch):
        K = item["labels"].size(0)
        labels[i, :K] = item["labels"]
        roles[i, :K] = item["roles"]
        sent_len[i, :K] = item["sent_len"]
        position[i, :K] = item["position"]
        sent_mask[i, :K] = 1.0

        is_question[i, :K] = item["is_question"]
        disc_marker[i, :K] = item["disc_marker"]

        sent_spans_batch.append(item["sent_spans"])

    # speaker_change：当前句和上一句角色是否变化
    speaker_change = torch.zeros_like(roles)
    speaker_change[:, 1:] = (roles[:, 1:] != roles[:, :-1]).long()

    # run_len_same_speaker：同一说话人连续长度
    run_len_same_speaker = torch.zeros_like(roles)
    for b in range(B):
        run = 0
        for k in range(K_max):
            if sent_mask[b, k] == 0:
                run_len_same_speaker[b, k] = 0
                continue
            if k == 0:
                run = 1
            else:
                if roles[b, k] == roles[b, k - 1]:
                    run += 1
                else:
                    run = 1
            run_len_same_speaker[b, k] = run

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "sent_spans": sent_spans_batch,
        "sent_mask": sent_mask,
        "labels": labels,
        "roles": roles,
        "sent_len": sent_len,
        "position": position,
        "is_question": is_question,
        "disc_marker": disc_marker,
        "speaker_change": speaker_change,
        "run_len_same_speaker": run_len_same_speaker,
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
