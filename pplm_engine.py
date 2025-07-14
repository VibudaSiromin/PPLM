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
def perturb_past(model, input_ids, past, loss_fn, steps=3, step_size=0.01):
    device = input_ids.device

    # Get input embeddings instead of using input_ids (integers don't support gradients)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds.retain_grad()                # So .grad is not None
    inputs_embeds.requires_grad_()             # Enable gradient tracking

    for step in range(steps):
        # Forward pass
        outputs = model(
            inputs_embeds=inputs_embeds,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True
        )

        logits = outputs.logits
        hidden = outputs.hidden_states[-1]

        # Compute loss and backward
        loss = loss_fn(logits, hidden)
        model.zero_grad()
        if inputs_embeds.grad is not None:
            inputs_embeds.grad.zero_()
        loss.backward()

        # Get gradients and apply perturbation
        grads = inputs_embeds.grad
        if grads is None:
            raise RuntimeError("Gradient is None. Ensure requires_grad is True and computation graph is connected.")

        # Gradient ascent on embeddings
        grad_direction = step_size * grads / (grads.norm() + 1e-10)
        inputs_embeds = (inputs_embeds + grad_direction).detach()
        inputs_embeds.requires_grad_()
        inputs_embeds.retain_grad()

    # Final forward pass to get modified logits
    outputs = model(
        inputs_embeds=inputs_embeds,
        past_key_values=past,
        use_cache=True,
        output_hidden_states=True
    )

    return outputs.logits


# === Full generation loop ===
@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    bow_vec=None,
    disc_model=None,
    loss_fn=None,
    steps=10,
    step_size=0.03,
    max_len=50
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

def loss_fn(logits, hidden, bow_vec=None, disc_model=None):
    losses = []

    if bow_vec is not None:
        # BoW Loss
        probs = F.softmax(logits, dim=-1)
        bow_probs = (probs * bow_vec.to(probs.device)).sum(dim=-1)
        bow_loss = -torch.log(bow_probs + 1e-12).mean()
        losses.append(bow_loss)

    if disc_model is not None:
        # Discriminator Loss
        with torch.no_grad():
            pooled_hidden = hidden[:, -1, :]  # Last token's hidden state
        pred = disc_model(pooled_hidden)
        target = torch.tensor([1], dtype=torch.long).to(logits.device)  # class label
        disc_loss = F.cross_entropy(pred, target)
        losses.append(disc_loss)

    return sum(losses)
