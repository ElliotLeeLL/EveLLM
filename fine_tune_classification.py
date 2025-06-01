import torch
import tiktoken
from pathlib import Path
from torch.utils.data import DataLoader
import time

from configuration import model_configs
from model import EveLLMModel
from utils.model_utils import *
from utils.gpt_download import download_and_load_gpt2
from dataset.eve_dataset import SpamDataset


def train_classifier_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs,eval_freq, eval_iter
):
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    examples_num, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_num += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device,eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        train_accuracy = calc_accuracy_loader(model=model, data_loader=train_loader, device=device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(model=model, data_loader=val_loader, device=device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy * 100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
    return train_losses, val_losses, train_accuracies, val_accuracies, examples_num

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


# Replace the output layer
num_classes = 2
model.out_head = torch.nn.Linear(
    config["emb_dim"], num_classes
)
model.to(device)

# Froze parameters for all layers except the last transformer block and the output layer
for param in model.parameters():
    param.requires_grad = False
for param in model.transformer_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

tokenizer = tiktoken.get_encoding("gpt2")
inputs = torch.tensor(tokenizer.encode("Do you have time")).unsqueeze(0)

with torch.no_grad():
    outputs = model(inputs.to(device))

extracted_path = "sms_spam_collection"

train_dataset = SpamDataset(
    csv_file=Path(extracted_path) / "train.csv",
    max_length=None,
    tokenizer=tokenizer
)

val_dataset = SpamDataset(
    csv_file=Path(extracted_path) / "validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

test_dataset = SpamDataset(
    csv_file=Path(extracted_path) / "test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

num_workers = 0
batch_size = 8

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)

# Train the model for classification tasks
start_time = time.time()
torch.manual_seed(43)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5
train_losses, val_losses, train_accuracies, val_accuracies, examples_num = train_classifier_simple(
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    device = device,
    optimizer = optimizer,
    num_epochs = num_epochs,
    eval_freq = 50,
    eval_iter = 5
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

