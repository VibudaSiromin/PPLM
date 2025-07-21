import torch
import torch.nn.functional as F
from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor

def loss_fn(logits, hidden, bow_vec=None, disc_model=None, disc_target=None):
    losses = []

    # === BoW loss ===
    if bow_vec is not None:
        probs = F.softmax(logits, dim=-1)
        bow_vec = bow_vec.to(probs.dtype)

        bow_probs = (probs * bow_vec).sum(dim=-1)

        print("[DEBUG] BoW probs stats:")
        print("  min:", bow_probs.min().item())
        print("  max:", bow_probs.max().item())
        print("  mean:", bow_probs.mean().item())

        bow_probs = torch.clamp(bow_probs, min=1e-4)  # safer lower bound
        bow_loss = -torch.log(bow_probs).mean()

        print(f"[DEBUG] BoW loss: {bow_loss.item()}")
        losses.append(0.2 * bow_loss)

    # === Discriminator loss ===
    if disc_model is not None and disc_target is not None:
        pooled_hidden = hidden[:, -1, :].to(dtype=torch.float32)

        if torch.isnan(pooled_hidden).any() or torch.isinf(pooled_hidden).any():
            raise ValueError("[ERROR] pooled_hidden contains NaN or Inf")

        pred = disc_model(pooled_hidden)

        if torch.isnan(pred).any() or torch.isinf(pred).any():
            raise ValueError("[ERROR] Discriminator output contains NaN or Inf")

        target = torch.tensor([disc_target] * pred.size(0), dtype=torch.long).to(pred.device)

        disc_loss = F.cross_entropy(pred, target)

        if torch.isnan(disc_loss) or torch.isinf(disc_loss):
            raise ValueError("[ERROR] Discriminator loss is NaN/Inf")

        print(f"[DEBUG] Discriminator loss: {disc_loss.item()}")
        losses.append(disc_loss)

    # === Fallback dummy loss if none used ===
    if len(losses) == 0:
        return logits.mean()

    total_loss = sum(losses)
    print(f"[DEBUG] Total loss: {total_loss.item()}")
    return total_loss

def perturb_past(model, input_ids, past, loss_fn, steps=3, step_size=0.00005):
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
        print(f"[DEBUG] Step {step+1}: Loss={loss.item():.6f}, Grad norm={grad_norm.item():.6f}")

        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            raise ValueError("NaN or Inf in gradients.")

        # Optional: clip large gradients
        max_norm = 5.0
        if grad_norm > max_norm:
            print(f"[DEBUG] Clipping gradient from {grad_norm.item()} to {max_norm}")
            grads = grads * (max_norm / grad_norm)

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
             disc_target=1, steps=1, step_size=0.0001, max_len=100,
             top_p=0.9, top_k=50, temperature=1.0):

    print("[DEBUG] disc_target:", disc_target)
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids=input_ids, use_cache=True, output_hidden_states=True, return_dict=True)
    past = outputs.past_key_values

    logits_processor = LogitsProcessorList([
        RepetitionPenaltyLogitsProcessor(penalty=1.2)
    ])

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

        # === Penalize [INST]-like tokens === 
        bad_token_ids = [518, 25580, 29962]  # ▁[, INST, ]
        emoji_token_seqs = [[29871, 243, 162, 155, 141], [29871, 243, 162, 155, 135], [29871, 243, 162, 155, 132], [29871, 243, 162, 155, 134], [29871, 243, 162, 168, 182], [29871, 243, 162, 143, 162], [29871, 243, 162, 145, 140], [29871, 243, 162, 155, 165], [29871, 243, 162, 155, 151], [29871, 243, 162, 155, 176], [29871, 243, 162, 149, 151], [29871, 243, 162, 168, 189], [29871, 243, 162, 155, 161], [29871, 243, 162, 155, 164], [29871, 243, 162, 155, 163], [29871, 243, 162, 151, 168], [29871, 243, 162, 149, 165], [29871, 243, 162, 155, 171], [29871, 243, 162, 155, 180], [29871, 243, 162, 155, 179], [29871, 243, 162, 148, 131], [29871, 243, 162, 174, 166], [29871, 229, 160, 167, 30598], [29871, 243, 162, 155, 144], [29871, 243, 162, 149, 152], [29871, 243, 162, 149, 153], [29871, 243, 162, 155, 155], [29871, 243, 162, 149, 154], [29871, 243, 162, 155, 181], [29871, 243, 162, 155, 178], [29871, 243, 162, 155, 182], [29871, 243, 162, 155, 177], [29871, 243, 162, 167, 178], [29871, 243, 162, 156, 133], [29871, 243, 162, 155, 147], [29871, 243, 162, 156, 134], [29871, 243, 162, 155, 186], [29871, 243, 162, 155, 153], [29871, 243, 162, 155, 141], [29871, 243, 162, 155, 135], [29871, 243, 162, 155, 132], [29871, 243, 162, 155, 134], [29871, 243, 162, 168, 182], [29871, 243, 162, 143, 162], [29871, 243, 162, 145, 140], [29871, 229, 159, 171], [29871, 243, 162, 155, 137], [29871, 243, 162, 167, 154], [29871, 243, 162, 149, 174], [29871, 243, 162, 155, 189], [29871, 243, 162, 145, 139], [29871, 229, 155, 131, 30598], [29871, 243, 162, 155, 190], [29871, 243, 162, 149, 153], [29871, 243, 162, 167, 154], [29871, 243, 162, 143, 139], [29871, 229, 155, 131, 30598], [29871, 243, 162, 174, 133], [29871, 243, 162, 143, 186], [29871, 243, 162, 143, 187], [29871, 243, 162, 152, 141, 30598], [29871, 243, 162, 145, 147], [29871, 243, 162, 174, 185], [29871, 243, 162, 150, 153], [29871, 243, 162, 149, 174], [29871, 243, 162, 170, 155], [29871, 243, 162, 144, 131], [29871, 243, 162, 149, 147], [29871, 243, 162, 143, 162], [29871, 243, 162, 156, 143], [29871, 243, 162, 151, 168], [29871, 243, 162, 149, 173], [29871, 243, 162, 148, 144], [29871, 229, 157, 164], [29871, 243, 162, 157, 131], [29871, 243, 162, 145, 178], [29871, 243, 162, 143, 135], [29871, 243, 162, 145, 184], [29871, 243, 162, 146, 142, 30598], [29871, 243, 162, 170, 154], [29871, 243, 162, 149, 168], [29871, 243, 162, 145, 192], [29871, 243, 162, 167, 160], [29871, 243, 162, 174, 185], [29871, 229, 159, 171], [29871, 243, 162, 152, 141, 30598], [29871, 243, 162, 143, 135], [29871, 243, 162, 149, 164], [29871, 243, 162, 167, 160], [29871, 243, 162, 143, 136], [29871, 243, 162, 170, 155], [29871, 243, 162, 158, 164, 30598], [29871, 243, 162, 145, 147], [29871, 243, 162, 174, 133], [29871, 243, 162, 144, 134], [29871, 243, 162, 151, 137], [29871, 243, 162, 173, 135], [29871, 243, 162, 150, 155], [29871, 243, 162, 149, 153], [29871, 243, 162, 149, 152], [29871, 243, 162, 149, 154], [29871, 243, 162, 155, 144], [29871, 229, 160, 167, 30598], [29871, 243, 162, 149, 161], [29871, 243, 162, 149, 155], [29871, 243, 162, 155, 155], [29871, 243, 162, 143, 186], [29871, 243, 162, 174, 185], [29871, 243, 162, 143, 187], [29871, 243, 162, 149, 147], [29871, 243, 162, 168, 179], [29871, 243, 162, 167, 154], [29871, 243, 162, 145, 131], [29871, 243, 162, 167, 172], [29871, 243, 162, 145, 140], [29871, 243, 162, 143, 162], [29871, 243, 162, 145, 139], [29871, 229, 159, 171], [29871, 243, 162, 155, 177, 30722, 243, 162, 149, 171], [29871, 243, 162, 149, 174], [29871, 243, 162, 145, 141], [29871, 243, 162, 150, 169], [29871, 243, 162, 145, 132], [29871, 243, 162, 144, 176], [29871, 243, 162, 145, 163], [29871, 243, 162, 143, 163], [29871, 243, 162, 157, 131], [29871, 243, 162, 155, 181], [29871, 243, 162, 156, 133], [29871, 243, 162, 156, 134], [29871, 243, 162, 170, 155], [29871, 243, 162, 143, 187], [29871, 243, 162, 174, 185], [29871, 243, 162, 143, 190], [29871, 243, 162, 144, 134], [29871, 243, 162, 143, 194], [29871, 243, 162, 173, 183], [29871, 243, 162, 170, 155, 30722, 31135, 30598], [29871, 243, 162, 150, 153], [29871, 243, 162, 173, 186], [29871, 243, 162, 151, 136], [29871, 229, 155, 152], [29871, 243, 162, 150, 157], [29871, 243, 162, 149, 174], [29871, 243, 162, 143, 186], [29871, 243, 162, 170, 191], [29871, 243, 162, 143, 139], [29871, 229, 159, 171], [29871, 243, 162, 174, 170], [29871, 243, 162, 152, 141, 30598], [29871, 243, 162, 144, 142], [29871, 229, 155, 131, 30598], [29871, 243, 162, 170, 155], [29871, 243, 162, 143, 194], [29871, 243, 162, 143, 189], [29871, 243, 162, 173, 183], [29871, 243, 162, 150, 155], [29871, 243, 162, 158, 132]]
        emoji_ids_flat = set(tid for seq in emoji_token_seqs for tid in seq)    
        
        for tid in bad_token_ids:
            if tid in emoji_ids_flat:
                continue  # skip emojis
            if tid < logits.shape[-1]:
                logits[0, tid] -= 5.0  # strong negative bias
        
        logits = torch.clamp(logits, -50, 50)  # avoid softmax overflow

        logits = logits_processor(input_ids, logits)

        # Top-k sampling
        if top_k > 0:
            top_k_vals, top_k_indices = torch.topk(logits, top_k, dim=-1)
            top_k_mask = torch.full_like(logits, float("-inf"))
            top_k_mask.scatter_(1, top_k_indices, top_k_vals)
            logits = top_k_mask

        # Top-p sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Early stopping if [INST] 
        decoded_tail = tokenizer.decode(input_ids[0][-10:], skip_special_tokens=True)
        if decoded_tail.count("[INST]") > 0 or input_ids.shape[1] > max_len:
            print("[Early Stop] Repeated [INST] or max length reached.")
            break

        if next_token.item() == tokenizer.eos_token_id:
            break

        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past = outputs.past_key_values

    return tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)



