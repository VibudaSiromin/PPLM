import torch
import torch.nn.functional as F

def loss_fn(logits, hidden, bow_vec=None, disc_model=None):
    losses = []
    
    if bow_vec is not None:
        probs = F.softmax(logits, dim=-1)
        bow_vec = bow_vec.to(probs.dtype)
        bow_probs = (probs * bow_vec).sum(dim=-1)
        bow_probs = torch.clamp(bow_probs, min=1e-12)
        bow_loss = -torch.log(bow_probs).mean()
        losses.append(0.1 * bow_loss)
    
    if disc_model is not None:
        pooled_hidden = hidden[:, -1, :].to(dtype=torch.float32)
        
        # Add input normalization
        pooled_hidden = torch.nn.functional.normalize(pooled_hidden, p=2, dim=-1)
        
        with torch.no_grad():  # First check discriminator stability
            test_pred = disc_model(pooled_hidden)
            if torch.isnan(test_pred).any():
                print("Discriminator produced NaN during test!")
                return torch.tensor(0.0, requires_grad=True)
        
        pred = disc_model(pooled_hidden)
        pred = torch.nan_to_num(pred, nan=0.0)  # Safety
        
        target = torch.tensor([1], dtype=torch.long).to(logits.device)
        disc_loss = F.cross_entropy(pred, target)
        losses.append(disc_loss)
    
    total_loss = sum(losses)
    return torch.nan_to_num(total_loss, nan=0.0)


def perturb_past(model, input_ids, past, loss_fn, steps=3, step_size=0.01):
    device = input_ids.device

    # Prepare embedding input and enable gradient
    inputs_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds = inputs_embeds.clone().detach().requires_grad_(True)

    # Make sure embedding layer allows grad
    model.get_input_embeddings().weight.requires_grad = True

    for step in range(steps):
        # Forward pass using embeddings
        outputs = model(
            inputs_embeds=inputs_embeds,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True
        )

        logits = outputs.logits
        hidden = outputs.hidden_states[-1]

        loss = loss_fn(logits, hidden)

        print(f"[Step {step+1}] Loss: {loss.item()}")
        print(f"[DEBUG] inputs_embeds.requires_grad: {inputs_embeds.requires_grad}")
        print(f"[DEBUG] logits.requires_grad: {logits.requires_grad}")
        print(f"[DEBUG] hidden.requires_grad: {hidden.requires_grad}")
        print(f"[DEBUG] loss.requires_grad: {loss.requires_grad}")

        if not loss.requires_grad:
            raise RuntimeError("Loss is not connected to graph. Check embedding and model layers.")

        # Backward pass
        model.zero_grad()
        if inputs_embeds.grad is not None:
            inputs_embeds.grad.zero_()
        loss.backward(retain_graph=True)

        grads = inputs_embeds.grad
        if grads is None:
            raise RuntimeError("Gradients not found on inputs_embeds.")

        # Update embeddings
        grad_direction = step_size * grads / (grads.norm() + 1e-10)
        inputs_embeds = (inputs_embeds + grad_direction).detach()
        inputs_embeds.requires_grad_()
        inputs_embeds.retain_grad()

    # Final forward pass
    final_outputs = model(
        inputs_embeds=inputs_embeds,
        past_key_values=past,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True
    )

    return final_outputs.logits

def generate(model, tokenizer, prompt, bow_vec=None, disc_model=None, loss_fn=None,
             steps=1, step_size=0.001, max_len=100, top_p=0.9, top_k=50, temperature=1.0):

    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_tokens = set(input_ids[0].tolist())

    outputs = model(input_ids=input_ids, use_cache=True, output_hidden_states=True, return_dict=True)
    past = outputs.past_key_values

    for _ in range(len(prompt.split()), max_len):
        logits = perturb_past(
            model=model,
            input_ids=input_ids,
            past=past,
            loss_fn=lambda l, h: loss_fn(l, h, bow_vec, disc_model),
            steps=steps,
            step_size=step_size
        )

        logits = logits[:, -1, :] / temperature

        # === Repetition Penalty ===
        for token_id in set(input_ids[0].tolist()):
            logits[0, token_id] *= 0.8  # penalize repeating same tokens

        # === Top-k Sampling ===
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            top_k_vals, top_k_indices = torch.topk(logits, top_k, dim=-1)
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, top_k_indices, top_k_vals)

        # === Top-p (Nucleus) Sampling ===
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)

        # === Early stopping on repeated '### Instruction' ===
        decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if decoded_text.count("### Instruction") > 1:
            print("[Early Stop] Detected repeated Instruction block.")
            break

        if next_token.item() == tokenizer.eos_token_id:
            break

        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past = outputs.past_key_values

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

