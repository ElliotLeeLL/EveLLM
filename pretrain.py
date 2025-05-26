import tiktoken
import torch
from configuration import config
from model import AdamLLMModel
from utils.model_utils import calc_loss_batch, evaluate_model, generate_and_print_sample, generate_and_print_top_k
from dataset.adam_dataset import create_dataloader

def train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        eval_freq,
        eval_iter,
        start_context,
        tokenizer
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d})"
                    f"Train loss {train_loss:.3f}"
                    f"Val loss {val_loss:.3f}"
                )
        generate_and_print_top_k(
            model, tokenizer, device, start_context
        )
        # generate_and_print_sample(
        #     model, tokenizer, device, start_context
        # )

    return train_losses, val_losses, track_tokens_seen

# Prepare the dataset
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text_data = f.read()
train_ratio = 0.90
split_idx = int(len(text_data) * train_ratio)
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader(
    train_data,
    batch_size=2,
    max_length=config["context_length"],
    stride=config["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader(
    val_data,
    batch_size=2,
    max_length=config["context_length"],
    stride=config["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)


torch.manual_seed(43)
tokenizer = tiktoken.get_encoding("gpt2")
model = AdamLLMModel(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004,
    weight_decay=0.1
)
num_epochs = 10
train_losses, val_losses, track_tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs,
    eval_freq=5, eval_iter=5, start_context="Life is beautiful", tokenizer=tokenizer
)
