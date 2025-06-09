import urllib.request
import zipfile
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tiktoken
from utils.model_utils import format_input_alpaca


class EveDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1:i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        #
        for entry in data:
            instruction_alpaca = format_input_alpaca(entry)
            response_text = f"\n\n### Response: \n{entry['output']}"
            full_text = instruction_alpaca + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, idx):
        return self.encoded_texts[idx]

    def __len__(self):
        return len(self.encoded_texts)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

def random_split(df, train_fraction, validation_fraction):
    df = df.sample(frac=1, random_state=43).reset_index(drop=True)
    train_end = int(len(df) * train_fraction)
    validation_end = train_end + int(len(df) * validation_fraction)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = EveDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

# Test code
# with open("../the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
#
# max_length = 4
# vocab_size = 50257
# output_dim = 256
#
# dataloader = create_dataloader(
#     raw_text,
#     batch_size=8,
#     max_length=max_length,
#     stride=max_length,
#     shuffle=True,
#     drop_last=True,
#     num_workers=0
# )
#
# data_iter = iter(dataloader)
#
# token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# inputs, targets = next(data_iter)
# print("The shape of inputs: ")
# print(inputs.shape)
# token_embedded = token_embedding_layer(inputs)
# print("The shape of embedded inputs: ")
# print(token_embedded.shape)
# pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)
# pos_token_embedded = pos_embedding_layer(torch.arange(max_length))
# print("The shape of embedded positional inputs: ")
# print(pos_token_embedded.shape)

# # Test code
# url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# zip_path = "sms_spam_collection.zip"
# extracted_path = "sms_spam_collection"
# data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
# download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
#
#
# # Test code
# tokenizer = tiktoken.get_encoding("gpt2")
# train_dataset = SpamDataset(
#     csv_file="train.csv",
#     max_length=None,
#     tokenizer=tokenizer
# )
#
# val_dataset = SpamDataset(
#     csv_file="validation.csv",
#     max_length=train_dataset.max_length,
#     tokenizer=tokenizer
# )
#
# test_dataset = SpamDataset(
#     csv_file="test.csv",
#     max_length=train_dataset.max_length,
#     tokenizer=tokenizer
# )
#
# num_workers = 0
# batch_size = 8
#
# train_loader = DataLoader(
#     dataset=train_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers,
#     drop_last=True
# )
# val_loader = DataLoader(
#     dataset=val_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers,
#     drop_last=True
# )
# test_loader = DataLoader(
#     dataset=test_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers,
#     drop_last=True
# )

