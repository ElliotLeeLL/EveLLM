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
from dataset.eve_dataset import SpamDataset

if __name__ == "__main__":
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
    batch_size = 32

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
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        num_epochs=num_epochs,
        eval_freq=50,
        eval_iter=5
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_tensor = torch.linspace(0, examples_num, len(train_losses))
    plot_values(
        epochs_tensor,
        examples_tensor,
        train_losses,
        val_losses,
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_accuracies))
    examples_tensor = torch.linspace(0, examples_num, len(train_accuracies))
    plot_values(
        epochs_tensor,
        examples_tensor,
        train_accuracies,
        val_accuracies,
        label="accuracy",
    )

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # Predict messages
    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )
    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )
    label1 = classify_review(
        text_1,
        model,
        tokenizer,
        device,
        max_length=train_dataset.max_length,
    )
    label2 = classify_review(
        text_2,
        model,
        tokenizer,
        device,
        max_length=train_dataset.max_length,
    )

    print("Label1: ", label1)
    print("Label2: ", label2)

    torch.save(model.state_dict(), Path("result_models") / "review_classifier.pth")
    # model_state_dict = torch.load("review_classifier.pth, map_location=device")
    # model.load_state_dict(model_state_dict)