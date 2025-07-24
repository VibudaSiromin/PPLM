import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

fileName = 'Texts/Angry_older_Disc.json'

# Load JSON file (replace 'generated_texts.json' with your file path)
with open(fileName, 'r') as f:
    data = json.load(f)  # Assumes JSON contains a list of sentences

# Initialize metrics storage
bleu_scores = []
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
perplexities = []

# Initialize ROUGE scorer
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Load your LLaMA-2-7B model and tokenizer
model_name = "Vibuda/llama_trained"  # Your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Compute metrics for each sentence
for sentence in data:
    # Tokenize for BLEU (compare with itself if no reference exists)
    ref = [sentence.split()]
    hyp = sentence.split()
    bleu = sentence_bleu(ref, hyp)
    bleu_scores.append(bleu)

    # Compute ROUGE (compare with itself)
    rouge_scores = rouge_scorer.score(sentence, sentence)
    rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
    rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
    rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

    # Compute Perplexity
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    perplexities.append(perplexity)

# Compute averages and standard errors
def mean_and_std_err(scores):
    mean = np.mean(scores)
    std_err = np.std(scores) / np.sqrt(len(scores))
    return mean, std_err

bleu_mean, bleu_err = mean_and_std_err(bleu_scores)
rouge1_mean, rouge1_err = mean_and_std_err(rouge1_scores)
rouge2_mean, rouge2_err = mean_and_std_err(rouge2_scores)
rougeL_mean, rougeL_err = mean_and_std_err(rougeL_scores)
perp_mean, perp_err = mean_and_std_err(perplexities)

# Print results
print("=== Evaluation Metrics ===", fileName)
print(f"BLEU: {bleu_mean:.4f} ± {bleu_err:.4f}")
print(f"ROUGE-1: {rouge1_mean:.4f} ± {rouge1_err:.4f}")
print(f"ROUGE-2: {rouge2_mean:.4f} ± {rouge2_err:.4f}")
print(f"ROUGE-L: {rougeL_mean:.4f} ± {rougeL_err:.4f}")
print(f"Perplexity: {perp_mean:.4f} ± {perp_err:.4f}")
