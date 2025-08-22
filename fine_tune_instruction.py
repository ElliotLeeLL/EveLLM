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


if __name__ == "__main__":
    # Create the model with a config
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
    # replace_linear_with_lora(model, rank=16, alpha=16, config=config)
    model.to(device)

    # Froze parameters for all layers except the last transformer block and the output layer
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.transformer_blocks[-1].parameters():
    #     param.requires_grad = True
    # for param in model.final_norm.parameters():
    #     param.requires_grad = True
    # for param in model.out_head.parameters():
    #     param.requires_grad = True
    #
    # inputs = torch.tensor(tokenizer.encode("Do you have time")).unsqueeze(0)
    # with torch.no_grad():
    #     outputs = model(inputs.to(device))

    # Prepare datasets
    tokenizer_file_path = Path("Qwen3-0.6B") / "tokenizer.json"
    tokenizer = Qwen3Tokenizer(
        str(tokenizer_file_path),
        add_generation_prompt=True,
        add_thinking=False,
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
    train_data = train_data[:85]
    val_data = val_data[:5]
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

    # Fine tune the model
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=8
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=8
        )
    print("The original training loss:", train_loss)
    print("The original validation loss:", val_loss)

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    num_epochs = 2

    # train_losses, val_losses, tokens_seen = [], [], [0]
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, config=config,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input_alpaca(val_data[0]), tokenizer=tokenizer
    )
    end_time = time.time()

    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Plot diagram for losses
    training_loss_file_path = Path("result_data") / f"eve_llm_qwen3_instruction_loss_{datetime.now().strftime('%Y%m%d%H%M')}.json"
    with open(training_loss_file_path, "w") as file:
        json.dump(str(training_loss_file_path), file, indent=4)
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_tensor = torch.linspace(0, tokens_seen[-1], len(train_losses))
    plot_values(
        epochs_tensor,
        examples_tensor,
        train_losses,
        val_losses,
    )

    save_model(model, config)

    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input_alpaca(entry)

        token_ids = generate_top_k(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=512,
            context_size=config["context_length"],
            eos_id=tokenizer.eos_token_id,
            temperature=0.6,
            top_k=20,
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len("<|im_start|>user\n")+len(input_text)+len("<|im_end|>\n"):]
            .replace("### Response:", "")
            .strip()
        )
        response_text = response_text.split("</think>\n\n", 1)[-1]
        response_text = response_text.removesuffix("<|im_end|>")
        test_data[i]["model_response"] = response_text

    result_data_path = Path("result_data") / f"instruction_data_with_response_{model_name}_{datetime.now().strftime('%Y%m%d%H%M')}.json"
    with open(result_data_path, "w") as file:
        json.dump(test_data, file, indent=4)

    # model_state_dict = torch.load("eve_llm_instruction.pth", map_location=device)
    # model.load_state_dict(model_state_dict)