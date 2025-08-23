import argparse
import json
from datetime import datetime

import torch
import tiktoken
from pathlib import Path
from torch.utils.data import DataLoader
import time
from functools import partial
from tqdm import tqdm
from safetensors.torch import load_file

from configuration import model_configs_qwen3
from model import EveLLMModel
from tokenizer.llama_tokenizer import Tokenizer
from tokenizer.qwen_tokenizer import Qwen3Tokenizer
from utils.model_utils import *
from utils.diagram_utils import *
from utils.gpt_download import download_and_load_gpt2
from dataset.eve_dataset import InstructionDataset, custom_collate_fn
from instruction_dataset_download import download_and_load_file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(43)
model_name = "eve-llm-qwen3-0P6B"
config = model_configs_qwen3[model_name]
model = EveLLMModel(config)

# Load weight into the model
combined_weights = {}
weights_path = Path("Qwen3-0.6B") / "model.safetensors"
current_weights = load_file(weights_path)
combined_weights.update(current_weights)
load_weights_into_eve_llm_qwen3(model, config, combined_weights)
# replace_linear_with_lora(model, rank=16, alpha=16, config=config)
model.to(device)
model.to(device).eval()

# Prepare datasets
tokenizer_file_path = Path("Qwen3-0.6B") / "tokenizer.json"
tokenizer = Qwen3Tokenizer(
    str(tokenizer_file_path),
    add_thinking=False,
    add_generation_prompt=True
)
file_path = Path("instruction_data") / "instruction_data.json"
url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
data = download_and_load_file(file_path, url)
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = int(len(data) - train_portion - test_portion)

train_data = data[:train_portion]
val_data = data[train_portion + test_portion:]
test_data = data[train_portion:train_portion + test_portion]

# # Test code
# train_data = train_data[:85]
# val_data = val_data[:5]
test_data = test_data[:10]

num_workers = 0
batch_size = 4

custom_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=config["context_length"],
)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate_fn,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers
)
val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)
test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input_alpaca(entry)

    token_ids = generate_top_k(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=config["context_length"],
        eos_id=tokenizer.eos_token_id
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len("<|im_start|>user\n")+len(input_text)+len("<|im_end|>\n")::]
        .replace("### Response:", "")
        .strip()
    )
    response_text = response_text.removeprefix("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    response_text = response_text.removesuffix("<|im_end|>")
    test_data[i]["model_response"] = response_text

result_data_path = Path(
    "result_data") / f"instruction_data_with_response_{model_name}_{datetime.now().strftime('%Y%m%d%H%M')}.json"
with open(result_data_path, "w") as file:
    json.dump(test_data, file, indent=4)