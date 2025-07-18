import torch
import torch.nn.functional as F

def loss_fn(logits, hidden, bow_vec=None, disc_model=None, disc_target=None):
    losses = []

    if bow_vec is not None:
        probs = F.softmax(logits, dim=-1)
        bow_vec = bow_vec.to(probs.dtype)
        bow_probs = (probs * bow_vec).sum(dim=-1) + 1e-8
        bow_loss = -torch.log(bow_probs).mean()
        losses.append(0.5 * bow_loss)

    if disc_model is not None and disc_target is not None:
        pooled_hidden = hidden[:, -1, :].to(dtype=torch.float32)
        pred = disc_model(pooled_hidden)
        target = torch.tensor([disc_target] * pred.size(0), dtype=torch.long).to(pred.device)
        disc_loss = F.cross_entropy(pred, target)
        losses.append(disc_loss)

    if len(losses) == 0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    return sum(losses)

def perturb_past(model, input_ids, past, loss_fn, steps=3, step_size=0.01):
    device = input_ids.device

    inputs_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds = inputs_embeds.clone().detach().requires_grad_(True)

    for step in range(steps):
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

        if not loss.requires_grad:
            raise RuntimeError("Loss is not connected to graph.")

        model.zero_grad()
        if inputs_embeds.grad is not None:
            inputs_embeds.grad.zero_()

        loss.backward(retain_graph=True)

        grads = inputs_embeds.grad
        if grads is None:
            raise RuntimeError("No gradients on inputs_embeds.")

        grad_norm = grads.norm()
        if torch.isnan(grad_norm):
            raise ValueError("NaN in gradients.")

        inputs_embeds = (inputs_embeds + step_size * grads / (grad_norm + 1e-10)).detach()
        inputs_embeds.requires_grad_()
        inputs_embeds.retain_grad()

    final_outputs = model(
        inputs_embeds=inputs_embeds,
        past_key_values=past,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True
    )

    return final_outputs.logits


def generate(model, tokenizer, prompt, bow_vec=None, disc_model=None, loss_fn=None,
             disc_target=1, steps=1, step_size=0.001, max_len=100,
             top_p=0.9, top_k=50, temperature=1.0):

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
            loss_fn=lambda l, h: loss_fn(l, h, bow_vec, disc_model, disc_target),
            steps=steps,
            step_size=step_size
        )

        logits = logits[:, -1, :] / temperature

        # Repetition penalty
        for token_id in set(input_ids[0].tolist()):
            logits[0, token_id] *= 0.8

        # Top-k filtering
        if top_k > 0:
            top_k_vals, top_k_indices = torch.topk(logits, top_k)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(1, top_k_indices, top_k_vals)
            logits = mask

        # Top-p (nucleus) filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = float("-inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if decoded_text.count("### Instruction") > 1:
            print("[Early Stop] Detected repeated Instruction block.")
            break

        if next_token.item() == tokenizer.eos_token_id:
            break

        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past = outputs.past_key_values

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


