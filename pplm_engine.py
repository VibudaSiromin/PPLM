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
def perturb_past(model, input_ids, past_key_values, loss_fn, steps=1, step_size=0.03):
    """
    Safely apply PPLM-style perturbation using loss_fn on logits or hidden state.
    No retain_graph needed.
    """
    device = next(model.parameters()).device
    perturbed_logits = None

    for _ in range(steps):
        input_ids = input_ids.detach()
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )

        logits = outputs.logits[:, -1, :]  # shape: (1, vocab_size)
        hidden = outputs.hidden_states[-1][:, -1, :]  # shape: (1, hidden_size)

        # Track gradients on logits
        logits = logits.clone().detach().requires_grad_(True)

        # Compute loss from logits or hidden
        loss = loss_fn(logits.unsqueeze(1), hidden)

        # Backprop
        model.zero_grad()
        loss.backward()

        # Apply gradient
        grad = logits.grad
        if grad is not None:
            perturbed_logits = logits + step_size * grad
        else:
            perturbed_logits = logits

        perturbed_logits = perturbed_logits.detach()

    return perturbed_logits.unsqueeze(1)  # shape: (1, 1, vocab_size)

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
