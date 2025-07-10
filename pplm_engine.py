import torch
import torch.nn.functional as F

def bow_loss(logits, bow_vec):
    probs = F.softmax(logits[:, -1, :], dim=-1)
    return -torch.log((probs * bow_vec.to(probs.device)).sum(dim=-1) + 1e-12).mean()

def discrim_loss(hidden, disc_model):
    logits = disc_model(hidden)
    labels = torch.ones(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)

def perturb_past(model, input_ids, past_key_values, loss_fn, steps=3, step_size=0.02):
    grad_accum = None

    for _ in range(steps):
        input_ids = input_ids[:, -1:]  # last token
        model_outputs = model(input_ids=input_ids, past_key_values=past_key_values, output_hidden_states=True, use_cache=True)
        logits = model_outputs.logits
        hidden = model_outputs.hidden_states[-1][:, -1, :]  # last hidden token
        loss = loss_fn(logits, hidden)

        grads = torch.autograd.grad(loss, model_outputs.logits)[0]
        grad_accum = grads if grad_accum is None else grad_accum + grads
        loss.backward()

    return grad_accum

@torch.no_grad()
def generate(model, tokenizer, prompt, bow_vec=None, disc_model=None, steps=3, step_size=0.03, max_len=50):
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated = input_ids
    past = None

    for _ in range(max_len):
        input_ids = generated[:, -1:]
        outputs = model(input_ids=input_ids, past_key_values=past, use_cache=True, output_hidden_states=True)
        logits = outputs.logits
        past = outputs.past_key_values

        def loss_fn(logits, hidden):
            if bow_vec is not None:
                return bow_loss(logits, bow_vec)
            elif disc_model is not None:
                return discrim_loss(hidden, disc_model)
            else:
                return torch.tensor(0.0, requires_grad=True).to(device)

        if bow_vec is not None or disc_model is not None:
            grads = perturb_past(model, generated, past, loss_fn, steps, step_size)
            logits = logits + grads

        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)
