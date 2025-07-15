import torch
import torch.nn.functional as F

def loss_fn(logits, hidden, bow_vec=None, disc_model=None):
    losses = []

    if disc_model is not None:
        pooled_hidden = hidden[:, -1, :]
        pooled_hidden = pooled_hidden.to(dtype=torch.float32)  
        pred = disc_model(pooled_hidden)
        target = torch.tensor([1], dtype=torch.long).to(logits.device)
        disc_loss = F.cross_entropy(pred, target)
        losses.append(disc_loss)

    if disc_model is not None:
        pooled_hidden = hidden[:, -1, :]
        pred = disc_model(pooled_hidden)
        target = torch.tensor([1], dtype=torch.long).to(logits.device)
        disc_loss = F.cross_entropy(pred, target)
        losses.append(disc_loss)

    return sum(losses)

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
             steps=3, step_size=0.01, max_len=60):

    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # First forward pass to get past_key_values
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

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_token), dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        # Update past for next step
        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past = outputs.past_key_values

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
