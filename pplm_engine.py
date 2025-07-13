import torch
import torch.nn.functional as F

# === Loss for Bag-of-Words control ===
def bow_loss(logits, bow_vec):
    probs = F.softmax(logits[:, -1, :], dim=-1)  # Shape: (batch_size, vocab_size)
    bow_vec = bow_vec.to(probs.device)

    # dot product between probs and bow vector
    bow_prob = (probs * bow_vec).sum(dim=-1)
    return -torch.log(bow_prob + 1e-12).mean()

# === Loss for Discriminator control ===
def discrim_loss(hidden, disc_model):
    logits = disc_model(hidden)
    labels = torch.ones(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)

# === Perturb logits using gradients from bow/discriminator loss ===
def perturb_past(model, input_ids, past_key_values, loss_fn, steps=1, step_size=0.02):
    device = next(model.parameters()).device
    perturbed_logits = None

    for _ in range(steps):
        input_ids = input_ids.detach()  # No grads needed here
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True
        )

        logits = outputs.logits  # shape: (1, seq_len, vocab_size)
        hidden = outputs.hidden_states[-1][:, -1, :]  # last token's hidden state

        logits.retain_grad()  # make sure grads can flow through logits

        loss = loss_fn(logits, hidden)

        model.zero_grad()
        loss.backward()  

        grad = logits.grad  # shape: (1, seq_len, vocab_size)

        if grad is not None:
            # Apply gradient-based perturbation to logits (PPLM-style)
            perturbed_logits = logits + step_size * grad
        else:
            perturbed_logits = logits

    # Detach to prevent graph from leaking further
    return perturbed_logits.detach() if perturbed_logits is not None else logits.detach()


# === Full generation loop ===
def generate(model, tokenizer, prompt, bow_vec=None, disc_model=None,
             steps=3, step_size=0.03, max_len=50):
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated = input_ids
    past = None

    for _ in range(max_len):
        input_ids = generated[:, -1:]

        outputs = model(input_ids=input_ids, past_key_values=past,
                        use_cache=True, output_hidden_states=True)

        logits = outputs.logits
        past = outputs.past_key_values

        # Define loss function depending on control method
        def loss_fn(logits, hidden):
            if bow_vec is not None:
                return bow_loss(logits, bow_vec)
            elif disc_model is not None:
                return discrim_loss(hidden, disc_model)
            else:
                return torch.tensor(0.0, requires_grad=True).to(device)

        # Only perturb if using PPLM controls
        if bow_vec is not None or disc_model is not None:
            logits = perturb_past(model, generated, past, loss_fn, steps, step_size)

        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)
