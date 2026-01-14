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

from configuration import model_configs_gemma
from model import EveLLMModel
from tokenizer.gemma_tokenizer import GemmaTokenizer
from tokenizer.llama_tokenizer import Tokenizer
from tokenizer.qwen_tokenizer import Qwen3Tokenizer
from utils.KVCache import KVCache
from utils.model_utils import *
from utils.diagram_utils import *
from utils.gpt_download import download_and_load_gpt2
from dataset.eve_dataset import InstructionDataset, custom_collate_fn
from instruction_dataset_download import download_and_load_file


def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    with torch.no_grad():
        cache = KVCache(n_layers=model.config["n_layers"])
        model.reset_kv_cache()
        logits = model(token_ids, cache=cache)

        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

            if (eos_token_id is not None
                    and torch.all(next_token == eos_token_id)):
                break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)

            logits = model(next_token, cache=cache)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(43)
config = model_configs_gemma
model = EveLLMModel(config)
model.eval()

# Load weight into the model
combined_weights = {}
weights_path = Path("Gemma3-0.27B") / "model.safetensors"
current_weights = load_file(weights_path)
combined_weights.update(current_weights)
load_weights_into_eve_llm_gemma3(model, config, combined_weights)
model.to(device)

# Prepare datasets
tokenizer_file_path = Path("Gemma3-0.27B") / "tokenizer.json"
tokenizer = GemmaTokenizer(tokenizer_file_path=tokenizer_file_path)

prompt = "Give me a short introduction to large language models in 100 words."
# prompt = "Who is the most beautiful woman in the world?"
input_token_ids = tokenizer.encode(prompt)
text = tokenizer.decode(input_token_ids)
print(text)

input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

for token in generate_text_basic_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=500,
    eos_token_id=tokenizer.encode("<end_of_turn>")[-1]
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )
