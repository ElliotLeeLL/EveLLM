import torch
import tiktoken
from pathlib import Path
from torch.utils.data import DataLoader

from configuration import model_configs
from model import EveLLMModel
from utils.model_utils import *
from utils.gpt_download import download_and_load_gpt2
from dataset.eve_dataset import SpamDataset


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
print(inputs)
print(inputs.shape)

with torch.no_grad():
    outputs = model(inputs.to(device))
print(outputs)
print(outputs.shape)

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

train_accuracy = calc_accuracy_loader(
    train_loader, model, device, num_batches=10
)
val_accuracy = calc_accuracy_loader(
    val_loader, model, device, num_batches=10
)
test_accuracy = calc_accuracy_loader(
    test_loader, model, device, num_batches=10
)
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=5
    )
    test_loss = calc_loss_loader(
        test_loader, model, device, num_batches=5
    )
print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")