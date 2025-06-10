from datetime import datetime

import torch
import numpy as np
import tiktoken
from pathlib import Path

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
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(start_dim=0, end_dim=1), target_batch.flatten()
    )
    return loss

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
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
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

def load_weights_into_eve_llm_gpt(model, params):
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
        model.transformer_blocks[b].attention_layer.W_query.weight = assign(
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

def format_input_alpaca(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request. "
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

def format_input_phi3(entry):
    user_text = (
        f"<|user|>"
        f"\n{entry['instruction']} "
        f"{entry['input']}" if entry["input"] else ""
        f"\n\n"
    )
    return user_text

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
                    f"Ep {epoch + 1} (Step {global_step:06d}) "
                    f"Train loss {train_loss:.3f} "
                    f"Val loss {val_loss:.3f} "
                )
            # Clean the cache at the end of each batch
            torch.cuda.empty_cache()
        generate_and_print_top_k(
            model, tokenizer, device, start_context
        )
        # generate_and_print_sample(
        #     model, tokenizer, device, start_context
        # )

    return train_losses, val_losses, track_tokens_seen

def train_model(
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
                    f"Ep {epoch + 1} (Step {global_step:06d}) "
                    f"Train loss {train_loss:.3f} "
                    f"Val loss {val_loss:.3f} "
                )
            # Clean cache at the end of each batch
            torch.cuda.empty_cache()
        generate_and_print_top_k(
            model, tokenizer, device, start_context
        )
        # generate_and_print_sample(
        #     model, tokenizer, device, start_context
        # )

    return train_losses, val_losses, track_tokens_seen

def save_model(model, config):
    dic_name = Path("result_models")
    file_name = Path(f"{config['model_name']}_cl_{config['context_length']}_ed_{config['emb_dim']}_{datetime.now().strftime('%Y%m%d%H%M')}.pth")
    torch.save(model.state_dict(), dic_name / file_name)