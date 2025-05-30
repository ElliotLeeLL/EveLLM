import urllib.request
import zipfile
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tiktoken


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

class SpamDataset(Dataset):
    def __init__(
            self, csv_file, tokenizer, max_length=None, pad_token_id=50256
    ):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [
            tokenizer.encode(text)
            for text in self.data["text"]
        ]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad encoded texts to max length
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (
                    self.max_length - len(encoded_text)
            )
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if len(encoded_text) > max_length:
                max_length = len(encoded_text)
        return max_length


def download_and_unzip_spam_data(
        url, zip_path, extracted_path, data_file_path
):
    if data_file_path.exists():
        print(f"Data file already exists at {data_file_path}. Skipping download.")
        return

    # Ensure the directory for the zip file exists
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as f:
            f.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file, data_file_path)
    print(f"Data file downloaded and unzipped to {data_file_path}.")


def creat_balanced_classification_dataset(df):
    num_spam = df[df["label"] == "spam"].shape[0]
    ham_subset = df[df["label"] == "ham"].sample(
        num_spam, random_state=43
    )
    balanced_df = pd.concat([
        ham_subset,
        df[df["label"] == "spam"]
    ])
    return balanced_df


def map_classification_dataset(df):
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def random_split(df, train_fraction, validation_fraction):
    df = df.sample(frac=1, random_state=43).reset_index(drop=True)
    train_end = int(len(df) * train_fraction)
    validation_end = train_end + int(len(df) * validation_fraction)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


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

