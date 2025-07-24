import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer as rs
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from collections import Counter
from nltk import ngrams

fileName = "Angry_older_BoW.json"
# Load JSON file (replace with your file)
with open(fileName, 'r') as f:
    data = json.load(f)  # List of generated sentences

# Initialize metric storage
bleu_scores = []
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
perplexities = []

# For Dist-N
all_unigrams = []
all_bigrams = []
all_trigrams = []

# ROUGE scorer
rouge_scorer = rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Load your fine-tuned LLaMA model
model_name = "Vibuda/llama_trained"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Loop through each sentence
for sentence in data:
    tokens = sentence.split()

    # Collect for Dist-N
    all_unigrams.extend(tokens)
    all_bigrams.extend(list(ngrams(tokens, 2)))
    all_trigrams.extend(list(ngrams(tokens, 3)))

    # BLEU: sentence compared with itself (or use reference if available)
    ref = [tokens]
    hyp = tokens
    bleu = sentence_bleu(ref, hyp)
    bleu_scores.append(bleu)

    # ROUGE: self comparison (or use reference if available)
    rouge_scores = rouge_scorer.score(sentence, sentence)
    rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
    rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
    rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

    # Perplexity
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    perplexities.append(perplexity)

# Compute Dist-N
def compute_distinct_ngrams(all_ngrams):
    total = len(all_ngrams)
    unique = len(set(all_ngrams))
    return unique / total if total > 0 else 0.0

dist1 = compute_distinct_ngrams(all_unigrams)
dist2 = compute_distinct_ngrams(all_bigrams)
dist3 = compute_distinct_ngrams(all_trigrams)

# Average & Std Err
def mean_and_std_err(scores):
    mean = np.mean(scores)
    std_err = np.std(scores) / np.sqrt(len(scores))
    return mean, std_err

bleu_mean, bleu_err = mean_and_std_err(bleu_scores)
rouge1_mean, rouge1_err = mean_and_std_err(rouge1_scores)
rouge2_mean, rouge2_err = mean_and_std_err(rouge2_scores)
rougeL_mean, rougeL_err = mean_and_std_err(rougeL_scores)
perp_mean, perp_err = mean_and_std_err(perplexities)

# Final Output
print("=== Evaluation Metrics ===")
print(f"BLEU: {bleu_mean:.4f} ± {bleu_err:.4f}")
print(f"ROUGE-1: {rouge1_mean:.4f} ± {rouge1_err:.4f}")
print(f"ROUGE-2: {rouge2_mean:.4f} ± {rouge2_err:.4f}")
print(f"ROUGE-L: {rougeL_mean:.4f} ± {rougeL_err:.4f}")
print(f"Perplexity: {perp_mean:.4f} ± {perp_err:.4f}")
print(f"Dist-1 (Unigrams): {dist1:.4f}")
print(f"Dist-2 (Bigrams): {dist2:.4f}")
print(f"Dist-3 (Trigrams): {dist3:.4f}")
