import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import pandas as pd

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
    else:
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        percent = int(count * block_size) * 100 / total_size

        speed = int(progress_size / (1024 * duration) / 1000) if duration else 0
        sys.stdout.write(
            f"\r{int(percent)}% | {progress_size / (1024 ** 2):.2f} MB "
            f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed"
        )
        sys.stdout.flush()

def download_and_extract_dataset(url, file_name, file_dictionary):
    if not os.path.exists(file_dictionary):
        os.makedirs(file_dictionary)
        if os.path.exists(file_name):
            os.remove(file_name)
        urllib.request.urlretrieve(url, Path(file_dictionary) / file_name, reporthook=reporthook)
        print("\nExtracting file...")
        with tarfile.open(Path(file_dictionary) / file_name, "r:gz") as tar:
            tar.extractall(path=Path(file_dictionary))
    else:
        print("File already exists")

def load_dataset_to_dataframe(base_path="aclImdb", labels=None):
    data_frames = []
    if labels is None:
        labels = {"pos": 1, "neg": 0}
    for subset in ["train", "test"]:
        for label in ("pos", "neg"):
            path = Path(base_path, subset, label)
            for file in sorted(path.iterdir()):
                with open(file, "r", encoding="utf-8") as f:
                    data_frames.append(pd.DataFrame({"text": [f.read()], "label": [labels[label]]}))
    df = pd.concat(data_frames, ignore_index=True)
    df = df.sample(frac=1, random_state=43).reset_index(drop=True)
    return df

def partition_and_save(df, size=(35000, 5000, 10000)):
    df_shuffled = df.sample(frac=1, random_state=43).reset_index(drop=True)

    train_end = size[0]
    val_end = size[0] + size[1]

    train = df_shuffled.iloc[:train_end]
    val = df_shuffled.iloc[train_end:val_end]
    test = df_shuffled.iloc[val_end:]

    # TODO:
    dictionary_name = Path("aclImdb_data")
    train.to_csv(dictionary_name / "train.csv", index=False)
    val.to_csv(dictionary_name / "val.csv", index=False)
    test.to_csv(dictionary_name / "test.csv", index=False)

if __name__ == "__main__":
    dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    print("Downloading dataset...")
    download_and_extract_dataset(dataset_url, "aclImdb_v1.tar.gz", "aclImdb_data")
    print("Extracting dataset...")
    df = load_dataset_to_dataframe(str((Path("aclImdb_data") / "aclImdb")))
    print("Partitioning dataset...")
    partition_and_save(df)
