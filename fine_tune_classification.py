import torch
import tiktoken

from configuration import model_configs
from model import EveLLMModel
from utils.model_utils import load_weights_into_evellm_gpt, generate_top_k,text_to_token_ids, token_ids_to_text
from utils.gpt_download import download_and_load_gpt2


# Create a model with config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = "eve-llm-1558M"
config = model_configs[model_name]
model = EveLLMModel(config)
model.eval()

# Load weight into the model
settings, params = download_and_load_gpt2(
model_size="1558M", models_dir="gpt2"
)
load_weights_into_evellm_gpt(model, params)
model.to(device)

# Test the loaded model
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_top_k(
    model=model,
    idx=text_to_token_ids("Life is beautiful", tokenizer=tokenizer).to(device),
    max_new_tokens=25,
    context_size=config["context_length"],
    top_k=50,
    temperature=1.0
)

print(
    "Output text: ", token_ids_to_text(token_ids, tokenizer=tokenizer)
)