import torch
import numpy as np
import tiktoken

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat([idx, idx_next], dim=1)
    return idx

def evaluate_model(
    model, train_loader, val_loader, device, eval_iter
):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(
        logits, target_batch
    )
    return loss

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break
    return correct_predictions / num_examples

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch=input_batch, target_batch=target_batch, model=model, device=device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def generate_and_print_sample(
    model, tokenizer, device, start_context
):
    model.eval()
    context_size = model.position_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace('\n', ' '))
    model.train()

def generate_and_print_top_k(
    model, tokenizer, device, start_context,
    temperature=1.0, top_k=None, eos_id=None
):
    model.eval()
    context_size = model.position_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_top_k(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size, temperature=temperature, top_k=top_k, eos_id=eos_id
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace('\n', ' '))
    model.train()

def generate_top_k(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, top_idx = torch.topk(logits, int(top_k))
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keep_dim=True)
        if idx_next.item() == eos_id:
            break
        idx = torch.cat([idx, idx_next], dim=1)
    return idx

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
        "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_evellm_gpt(model, params):
    model.position_embedding.weight = assign(
        model.position_embedding.weight, params['wpe']
    )
    model.token_embedding.weight = assign(
        model.token_embedding.weight, params['wte']
    )

    for b in range(len(params["blocks"])):
        # Assign weights for attention layers
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        model.transformer_blocks[b].attention_layer.W_query.wight = assign(
            model.transformer_blocks[b].attention_layer.W_query.weight, q_w.T
        )
        model.transformer_blocks[b].attention_layer.W_key.weight = assign(
            model.transformer_blocks[b].attention_layer.W_key.weight, k_w.T
        )
        model.transformer_blocks[b].attention_layer.W_value.weight = assign(
            model.transformer_blocks[b].attention_layer.W_value.weight, v_w.T
        )

        # Assign biases for attention layers
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        model.transformer_blocks[b].attention_layer.W_query.bias = assign(
            model.transformer_blocks[b].attention_layer.W_query.bias, q_b
        )
        model.transformer_blocks[b].attention_layer.W_key.bias = assign(
            model.transformer_blocks[b].attention_layer.W_key.bias, k_b
        )
        model.transformer_blocks[b].attention_layer.W_value.bias = assign(
            model.transformer_blocks[b].attention_layer.W_value.bias, v_b
        )

        # Assign weights and bias for output layers
        model.transformer_blocks[b].attention_layer.output.weight = assign(
            model.transformer_blocks[b].attention_layer.output.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        model.transformer_blocks[b].attention_layer.output.bias = assign(
            model.transformer_blocks[b].attention_layer.output.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        # Assign weights and bias for mlp layers
        model.transformer_blocks[b].ff.layers[0].weight = assign(
            model.transformer_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        model.transformer_blocks[b].ff.layers[0].bias = assign(
            model.transformer_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        model.transformer_blocks[b].ff.layers[2].weight = assign(
            model.transformer_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        model.transformer_blocks[b].ff.layers[2].bias = assign(
            model.transformer_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        model.transformer_blocks[b].norm1.scale = assign(
            model.transformer_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"]
        )
        model.transformer_blocks[b].norm1.shift = assign(
            model.transformer_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"]
        )
        model.transformer_blocks[b].norm2.scale = assign(
            model.transformer_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"]
        )
        model.transformer_blocks[b].norm2.shift = assign(
            model.transformer_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"]
        )

    model.final_norm.scale = assign(
        model.final_norm.scale, params["g"]
    )
    model.final_norm.shift = assign(
        model.final_norm.shift, params["b"]
    )
    model.out_head.weight = assign(
        model.out_head.weight, params["wte"]
    )

def classify_review(
  text, model, tokenizer, device, max_length=None, pad_token_id=50256
):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.position_embedding.weight.shape[1]

    input_ids = input_ids[:min(max_length, supported_context_length)]

    input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_ids)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "ham"

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