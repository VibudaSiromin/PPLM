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
def perturb_past(
    model,
    input_ids,
    past,
    loss_fn,
    steps=3,
    step_size=0.03,
):
    device = input_ids.device
    grad_accumulator = None
    accumulated_hidden = None

    for _ in range(steps):
        # Get embeddings for the last token only
        last_token_id = input_ids[:, -1:]
        inputs_embeds = model.get_input_embeddings()(last_token_id)
        inputs_embeds.requires_grad_(True)

        # Forward with embeddings (instead of input_ids)
        outputs = model(
            inputs_embeds=inputs_embeds,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True  # this enables hidden_states
        )
        logits = outputs.logits
        hidden = outputs.hidden_states[-1]  # Last layer
        
        # Compute loss (BoW or discriminator)
        loss = loss_fn(logits, hidden)
        loss.backward()

        grads = inputs_embeds.grad  # Gradient w.r.t. embedding

        if grad_accumulator is None:
            grad_accumulator = grads.clone()
        else:
            grad_accumulator += grads

        # Clear memory
        model.zero_grad()
        torch.cuda.empty_cache()

    # Apply the final perturbation
    perturbed_embeds = inputs_embeds - step_size * grad_accumulator.sign()
    perturbed_embeds = perturbed_embeds.detach()

    # Final forward pass with perturbed embeddings
    outputs = model(inputs_embeds=perturbed_embeds, past_key_values=past, use_cache=True)
    return outputs.logits


# === Full generation loop ===
def generate(model, tokenizer, prompt, bow_vec=None, disc_model=None,
             steps=1, step_size=0.03, max_len=50):
    """
    Generates text using BoW and/or discriminator guidance (PPLM).
    """
    device = next(model.parameters()).device

    # Tokenize the initial prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids
    past = None

    for _ in range(max_len):
        input_ids = generated[:, -1:]  # last token only

        outputs = model(
            input_ids=input_ids,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True
        )

        logits = outputs.logits
        past = outputs.past_key_values

        # Define loss function
        def loss_fn(logits, hidden):
            if bow_vec is not None:
                return bow_loss(logits, bow_vec)
            elif disc_model is not None:
                return discrim_loss(hidden, disc_model)
            else:
                return torch.tensor(0.0, requires_grad=True).to(device)

        # Apply perturbation only if control method is selected
        if bow_vec is not None or disc_model is not None:
            logits = perturb_past(model, generated, past, loss_fn, steps, step_size)
        else:
            logits = logits[:, -1:, :]  # unperturbed logits

        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)
