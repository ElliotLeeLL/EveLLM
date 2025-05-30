from dataset.eve_dataset import download_and_unzip_spam_data, SpamDataset, creat_balanced_classification_dataset, random_split
from pathlib import Path
from torch.utils.data import DataLoader
import tiktoken
import pandas as pd


url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
extracted_path = "sms_spam_collection"
zip_path = Path("sms_spam_collection/sms_spam_collection.zip")
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["label", "text"]
)
balanced_df = creat_balanced_classification_dataset(df)
balanced_df["label"] = balanced_df["label"].map({"ham": 0, "spam": 1})
train_df, validation_df, test_df = random_split(
    balanced_df, 0.7, 0.1
)
train_df.to_csv(f"{extracted_path}/train.csv", index=None)
validation_df.to_csv(f"{extracted_path}/validation.csv", index=None)
test_df.to_csv(f"{extracted_path}/test.csv", index=None)

# Test code
tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = SpamDataset(
    csv_file=Path(extracted_path) / "train.csv",
    max_length=None,
    tokenizer=tokenizer
)

val_dataset = SpamDataset(
    csv_file=Path(extracted_path) / "validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

test_dataset = SpamDataset(
    csv_file=Path(extracted_path) / "test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

num_workers = 0
batch_size = 8

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)

print(len(train_loader))