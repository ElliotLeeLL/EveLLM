from huggingface_hub import login, hf_hub_download
import json

from transformers import AutoTokenizer

from configuration import model_configs_llama


tokenizer_file_path = hf_hub_download(
    repo_id="Qwen/Qwen3-0.6B-base",
    filename="tokenizer.json",
    local_dir="Qwen3-0.6B",
)

weights_file = hf_hub_download(
    repo_id="Qwen/Qwen3-0.6B-base",
    filename="model.safetensors",
    local_dir="Qwen3-0.6B"
)