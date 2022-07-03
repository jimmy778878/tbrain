import json
import copy
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def collate_for_training(batch):
    input_ids = []
    attention_mask = []
    labels = []

    for data in batch:
        input_ids.append(data["input_ids"])
        attention_mask.append(data["attention_mask"])
        labels.append(data["label"])
    
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return input_ids, attention_mask, labels


def collate_for_scoring(batch):
    utt_id = []
    hyp_id = []
    input_ids = []
    attention_mask = []
    mask_pos = []
    masked_token_id = []

    for data in batch:
        utt_id.append(data["utt_id"])
        hyp_id.append(data["hyp_id"])
        input_ids.append(data["input_ids"])
        attention_mask.append(data["attention_mask"])
        mask_pos.append(data["mask_pos"])
        masked_token_id.append(data["masked_token_id"])

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)

    return utt_id, hyp_id, input_ids, attention_mask, mask_pos, masked_token_id


def static_masking(
    token_seq: list[str], 
    tokenizer, 
    mlm_probability: float = 0.15, 
):
    cls_ids = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    sep_ids = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    mask_ids = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    input_ids = tokenizer.convert_tokens_to_ids(token_seq)
    input_ids = np.array(input_ids)
    label = copy.copy(input_ids)
    
    mlm_mask_pos = np.random.binomial(1, mlm_probability, len(input_ids))

    mask_token_pos = np.random.binomial(1, 0.8, len(input_ids))
    mask_token_pos = mask_token_pos & mlm_mask_pos
    mask_token_indices = mask_token_pos.nonzero()[0]
    input_ids[mask_token_indices] = mask_ids
    label[~mask_token_indices] = -100

    random_token_pos = np.random.binomial(1, 0.5, len(input_ids))
    random_token_pos = random_token_pos & mlm_mask_pos & ~mask_token_pos
    random_token_indices = random_token_pos.nonzero()[0]
    random_token_ids = np.random.randint(len(tokenizer), size=len(random_token_indices))
    input_ids[random_token_indices] = random_token_ids

    input_ids = np.append(np.append([cls_ids], input_ids), [sep_ids])
    label = np.append(np.append([-100], label), [-100])
    attention_mask = [1] * len(input_ids)

    return [{
        "input_ids": torch.tensor(input_ids, dtype= torch.long),
        "label": torch.tensor(label, dtype= torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype= torch.long),
    }]


def one_by_one_masking(
    token_seq: list[str],
    tokenizer,
    utt_id: str = None,
    hyp_id: str = None,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    mask_token = tokenizer.mask_token

    token_seq = [cls_token] + token_seq + [sep_token]

    output = []
    for mask_pos in range(1, len(token_seq)-1):
        input_ids = tokenizer.convert_tokens_to_ids(
            token_seq[:mask_pos] + [mask_token] + token_seq[mask_pos+1:]
        )
        label = tokenizer.convert_tokens_to_ids(token_seq)
        label[~mask_pos] = -100
        attention_mask = [1] * len(token_seq)
        output.append({
            "utt_id": utt_id,
            "hyp_id": hyp_id,
            "input_ids": torch.tensor(input_ids, dtype= torch.long),
            "label": torch.tensor(label, dtype= torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype= torch.long),
            "mask_pos": mask_pos,
            "masked_token_id": label[mask_pos]
        })

    return output


def get_dataloader_for_mlm_bert(
    model: str,
    task: str,
    data_path: str,
    masking_strategy: str,
    batch_size: int,
    shuffle: bool = False,
    max_utt: int = None,
    n_best: int = None
):
    tokenizer = BertTokenizer.from_pretrained(model)

    if task == "training":
        ref_json = json.load(open(data_path, "r", encoding="utf-8"))
        all_utt_data = []
        for num_utt, ref_text in tqdm(enumerate(ref_json.values()), total=min(max_utt, len(ref_json.values()))):
            if max_utt == num_utt:
                break
            token_seq = tokenizer.tokenize(ref_text)
            if masking_strategy == "static":
                one_utt_data = static_masking(token_seq, tokenizer)
            elif masking_strategy == "one_by_one":
                one_utt_data = one_by_one_masking(token_seq, tokenizer)
            all_utt_data += one_utt_data
        dataset = MyDataset(all_utt_data)
        dataloader = DataLoader(dataset, batch_size, shuffle, collate_fn=collate_for_training)

    elif task == "scoring":
        hyp_json = json.load(open(data_path, "r", encoding="utf-8"))
        all_hyp_data = []
        for num_utt, (utt_id, hyps) in tqdm(enumerate(hyp_json.items()), total=min(max_utt, len(hyp_json.items()))):
            if max_utt > num_utt:
                break
            for num_hyp, (hyp_id, hyp_text) in enumerate(hyps.items()):
                if n_best == num_hyp:
                    break
                token_seq = tokenizer.tokenize(hyp_text)
                one_hyp_data = one_by_one_masking(token_seq, tokenizer, utt_id, hyp_id)
                all_hyp_data += one_hyp_data
        dataset = MyDataset(all_hyp_data)
        dataloader = DataLoader(dataset, batch_size, shuffle, collate_fn=collate_for_scoring)

    return dataloader