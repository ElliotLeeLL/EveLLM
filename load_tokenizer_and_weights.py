from huggingface_hub import login, hf_hub_download
import json

from transformers import AutoTokenizer

from configuration import model_configs_llama


configuration = model_configs_llama["LLAMA32_CONFIG_1B"]
login(configuration["HF_ACCESS_TOKEN"])

tokenizer_file_path = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-1B",
    filename="original/tokenizer.model",
    local_dir="Llama-3.2-1B",
)

weights_file = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-1B",
    filename="model.safetensors",
    local_dir="Llama-3.2-1B"
)