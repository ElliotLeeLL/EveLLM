import urllib.request
import zipfile
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    def __init__(
        self, csv_file, tokenizer, max_length=None, pad_token_id=50256
    ):
        return
    def __getitem__(self, index):
        return
    def __len__(self):
        return
    def _longest_encoded_length(self):
        return

def download_and_unzip_data(
    url, zip_path, extracted_path, data_file_path
):
    if data_file_path.exists():
        print(f"Data file already exists at {data_file_path}. Skipping download.")
        return

    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as f:
            f.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    
    original_file = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file, data_file_path)
    print(f"Data file downloaded and unzipped to {data_file_path}.")

def creat_balanced_classifacation_dataset(df):
    num_spam = df[df["label"] == "spam"].shape[0]
    ham_subset = df[df["label"] == "ham"].sample(
        num_spam, random_state=43
    )
    balanced_df = pd.concat([
        ham_subset,
        df[df["label"] == "spam"]
    ])
    return balanced_df
    
def map_classifacation_dataset(df):
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

# # Test code
# url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# zip_path = "sms_spam_collection.zip"
# extracted_path = "sms_spam_collection"
# data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"