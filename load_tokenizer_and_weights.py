import json
import os
from pathlib import Path
from safetensors.torch import  load_file
from huggingface_hub import hf_hub_download, snapshot_download, login

# To download Gemma3-270m-it weight from hugging face, you need to accept google's licensing terms and create a token on hugging face website
# login(token="PLEASE ENTER YOUR TOKEN HERE")

CHOOSE_MODEL = "270m"
USE_INSTRUCT_MODEL = True

if USE_INSTRUCT_MODEL:
    repo_id = f"google/gemma-3-{CHOOSE_MODEL}-it"
else:
    repo_id = f"google/gemma-3-{CHOOSE_MODEL}"

local_dir = Path(__file__).resolve().parent / "Gemma3-0.27B"

weight_file = hf_hub_download(
    repo_id=repo_id,
    filename="model.safetensors",
    local_dir=local_dir,
)

