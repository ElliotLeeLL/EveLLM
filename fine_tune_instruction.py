import torch
import tiktoken
from pathlib import Path
from torch.utils.data import DataLoader
import time

from configuration import model_configs
from model import EveLLMModel
from utils.model_utils import *
from utils.diagram_utils import *
from utils.gpt_download import download_and_load_gpt2
from dataset.eve_dataset import InstructionDataset, custom_collate_fn
from instruction_dataset_download import download_and_load_file

# Create a model with config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(43)
model_name = "eve-llm-355M"
config = model_configs[model_name]
model = EveLLMModel(config)
model.eval()

# Load weight into the model
settings, params = download_and_load_gpt2(
model_size="355M", models_dir="gpt2"
)
load_weights_into_evellm_gpt(model, params)

# Froze parameters for all layers except the last transformer block and the output layer
model.to(device)
for param in model.parameters():
    param.requires_grad = False
for param in model.transformer_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
for param in model.out_head.parameters():
    param.requires_grad = True

tokenizer = tiktoken.get_encoding("gpt2")
inputs = torch.tensor(tokenizer.encode("Do you have time")).unsqueeze(0)

with torch.no_grad():
    outputs = model(inputs.to(device))

# Prepare datasets
file_path = Path("instruction_data") / "instruction_data.json"
url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
data = download_and_load_file(file_path, url)
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = int(len(data) - train_portion - test_portion)

train_data = data[:train_portion]
val_data = data[train_portion + test_portion:]
test_data = data[train_portion:train_portion + test_portion]

train_portion = int()
num_workers = 0
batch_size = 8

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)
val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)
test_dataset = InstructionDataset(val_data, tokenizer)
test_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)