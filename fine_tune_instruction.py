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

# Create a model with config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

print(outputs)
