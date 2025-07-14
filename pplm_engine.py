import torch
import torch.nn.functional as F

# === BoW loss ===
def bow_loss(logits, bow_vec):
    probs = F.softmax(logits, dim=-1)
    bow_vec = bow_vec.to(probs.device)
    bow_probs = (probs * bow_vec).sum(dim=-1)
    return -torch.log(bow_probs + 1e-12).mean()

# === Discriminator loss ===
def discrim_loss(hidden, disc_model):
    logits = disc_model(hidden)
    labels = torch.ones(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)

# === Perturb logits ===
def perturb_past(model, generated, past, loss_fn, steps=5, step_size=0.04):
    device = generated.device

    grad_accumulator = None

    for _ in range(steps):
        input_ids = generated[:, -1:]
        input_embeds = model.get_input_embeddings()(input_ids)
        input_embeds.requires_grad_(True)

        outputs = model(
            inputs_embeds=input_embeds,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True
        )
        logits = outputs.logits
        hidden = outputs.hidden_states[-1]

        # === Compute loss ===
        loss = loss_fn(logits, hidden)

        # === Backprop ===
        loss.backward()

        # === Collect gradients ===
        grads = input_embeds.grad
        if grads is None:
            raise RuntimeError("Gradient is None. Did you forget to call .backward()?")

        grad_accumulator = grads.clone()

        # === Update embedding with gradient ===
        perturbation = step_size * grad_accumulator
        input_embeds = input_embeds + perturbation.detach()
        input_embeds.requires_grad_(True)

    # === Final forward pass after perturbation ===
    final_outputs = model(
        inputs_embeds=input_embeds,
        past_key_values=past,
        use_cache=True
    )
    return final_outputs.logits


# === Full generation loop ===
@torch.no_grad()
def generate(
    model, tokenizer, prompt, bow_vec=None, disc_model=None,
    steps=5, step_size=0.04, max_len=60
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Encode prompt ===
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids.clone()

    # === Generate tokens iteratively ===
    for _ in range(max_len):
        # 1. Forward pass to get past_key_values
        outputs = model(input_ids=generated, use_cache=True, output_hidden_states=True)
        past = outputs.past_key_values

        # 2. Apply perturbation to influence generation
        logits = perturb_past(
            model=model,
            generated=generated,
            past=past,
            loss_fn=lambda logits, hidden: loss_fn(logits, hidden, bow_vec, disc_model),
            steps=steps,
            step_size=step_size
        )

        # 3. Sample or take the top token
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        # 4. Append to generated sequence
        generated = torch.cat((generated, next_token), dim=1)

        # Stop if end-of-sequence
        if next_token.item() == tokenizer.eos_token_id:
            break

    # === Decode final text ===
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output_text
