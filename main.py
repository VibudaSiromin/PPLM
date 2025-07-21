import torch
import json
import gc

from models import Discriminator
from load_lm import load_lm
from bow_utils import load_bow_vector
from pplm_engine import generate, loss_fn as base_loss_fn

# === Choose base model ===
MODEL_NAME = "Vibuda/llama_trained" # "Qwen/Qwen1.5-7B-Chat" 

# === Load model in 4-bit ===
model, tokenizer = load_lm(MODEL_NAME)
device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

emoji_list =  ["😊", "😄", "😁", "😃", "🥳", "🌟", "🎉","😢", "😔", "😭", "💔", "🥺", "😞",
"😡", "😠","🔥", "💢","😨", "😱", "😰", "👀", "🫣","❤️", "😍", "💕", "💖", "😘", "💗",
"😲", "😯", "😳", "😮", "🤯","🙂", "😐", "🙃","😷", "😖",
"😊", "😄", "😁", "😃", "🥳", "🌟", "🎉", "✨", "😆", "🤗", "💫", "😺", "🎈", "☀️", "😻",
"💖", "🤗", "🌈", "☀️", "🫂", "🌷", "🌸", "🕊️", "🎐", "🫶", "📖", "💫", "🧘", "🍀", "💐",
"🌟", "🙌", "🔥", "💪", "👍", "⚡", "🚀", "🎯", "🌄", "🎵", "🏋️", "🧗", "💥", "🎽", "🤝",
"🫶", "✨", "🕊️", "🌄", "💡", "🤝", "🌅", "🧘", "🛡️", "🎐", "🫂", "🍃", "🔆", "🪄", "📘",
"💖", "💕", "💗", "😍", "❤️", "💞", "💘", "😘", "🌷", "🫶", "🌸", "💐", "🥰", "🤗", "🎀",
"🤩", "🎉", "🌟", "🎈", "✨", "😮‍💨", "💫", "🎊", "📦", "🎁", "🍭", "🎠", "🌠", "🚀", "😲",
"🙂", "🙃", "🧘", "🌸", "🫶", "🌻", "🍃", "🌿", "🪴", "🧘‍♂️", "📖", "🪷", "🔅", "☕", "📚",
"💫", "🌷", "🧼", "🌈", "✨", "🫧", "🕊️", "🍋", "☀️", "🧘", "🌿", "🌺", "🪴", "📘", "🛁"
]
 
# Print token IDs for each emoji
print("=== Emoji Token IDs ===")
emoji_ids = []
for emoji in emoji_list:
    token_ids = tokenizer.encode(emoji, add_special_tokens=False)
    emoji_ids.extend(token_ids)
    print(f"{emoji}: {token_ids}")

# Deduplicate the final list (some emojis map to multi-token)
emoji_ids = list(set(emoji_ids))
print("\n✅ Final emoji ID whitelist:", emoji_ids)

# Ensure the embedding layer allows gradient
model.get_input_embeddings().weight.requires_grad = True

# === GPU cleanup before generation ===
gc.collect()
torch.cuda.empty_cache()

# === Settings ===
USE_BOW = True
USE_DISC = False
TARGET_GROUP = "older"  # or "younger"
disc_target = 0 if TARGET_GROUP == "younger" else 1

question = "Why does nobody take me seriously no matter how loud I speak up?"
prompt = f"[INST] {question} [/INST]\nResponse:"

# === Load BoW ===
bow_vec = load_bow_vector(
    f"bow_{TARGET_GROUP}.json",
    tokenizer,
    expected_vocab_size=model.config.vocab_size
).to(device)

print(f"[BoW shape]: {bow_vec.shape} | model vocab size: {tokenizer.vocab_size}")

disc_model = Discriminator(hidden_size=model.config.hidden_size).to(device)

# === Load Discriminator ===
if USE_DISC:
    if "qwen" in MODEL_NAME:
        checkpoint = torch.load("discriminator_qwen.pt", map_location=device)
    elif "llama" in MODEL_NAME:
        checkpoint = torch.load("discriminator_llama.pt", map_location=device)
    else:
        raise ValueError(f"No checkpoint mapping defined for model: {MODEL_NAME}")
        
    disc_model.load_state_dict(checkpoint['model_state_dict'])
    disc_model.eval()
else:
    disc_model = None

print(f"[BoW vector non-zero entries]: {bow_vec.nonzero().shape[0]}")

if disc_model is not None:
    for name, param in disc_model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise ValueError(f"[ERROR] Discriminator weight '{name}' contains NaN or Inf")
    print("[DEBUG] Discriminator model loaded successfully.")

# === Generate ===
output = generate(
    model,
    tokenizer,
    prompt,
    bow_vec=bow_vec,
    disc_model=disc_model,
    loss_fn=base_loss_fn,
    disc_target=disc_target,
    steps=1,           
    step_size=0.00001,    
    max_len=500,
    top_k=50,
    top_p=0.9,          
    temperature=0.7
)

print("😊\n[Generated Text]")
print(output)

# === Save to file ===
output_file = f"Angry_{TARGET_GROUP}_USE_BOW_{USE_BOW}_USE_DISC_{USE_DISC}.txt"

# def force_unicode(text):
#     try:
#         # First try normal decoding
#         return str(text).encode('utf-8', 'strict').decode('utf-8')
#     except UnicodeError:
#         try:
#             # Fallback to surrogate escape for damaged unicode
#             return str(text).encode('utf-8', 'surrogateescape').decode('utf-8')
#         except UnicodeError:
#             # Final fallback - replace problematic characters
#             return str(text).encode('utf-8', 'replace').decode('utf-8')

# # Prepare content with proper newlines and separators
# content = []
# if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
#     content.append("\n\n")  # Add spacing if file exists

# content.extend([
#     f"\n=== Response ===\n{force_unicode(output)}\n",
#     "-" * 50 + "\n"
# ])

# # Write using binary mode with explicit UTF-8 encoding
# with open(output_file, "ab") as f:  # Note 'b' for binary mode
#     for part in content:
#         f.write(part.encode('utf-8'))

# print(f"\nResponse successfully saved with emoji preservation to {output_file}")
