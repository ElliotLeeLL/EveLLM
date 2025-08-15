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


def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if (eos_token_id is not None
                    and torch.all(next_token == eos_token_id)):
                break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(43)
model_name = "eve-llm-qwen3-0P6B"
config = model_configs_qwen3[model_name]
model = EveLLMModel(config)
model.eval()

# Load weight into the model
combined_weights = {}
weights_path = Path("Qwen3-0.6B") / "model.safetensors"
current_weights = load_file(weights_path)
combined_weights.update(current_weights)
load_weights_into_eve_llm_qwen3(model, config, combined_weights)
model.to(device)

# Prepare datasets
tokenizer_file_path = Path("Qwen3-0.6B") / "tokenizer.json"
tokenizer = Qwen3Tokenizer(
    str(tokenizer_file_path),
    add_generation_prompt=True,
    add_thinking=True,
)

prompt = "Give me a short introduction to large language models."
input_token_ids = tokenizer.encode(prompt)
text = tokenizer.decode(input_token_ids)
print(text)

input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

for token in generate_text_basic_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=500,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )