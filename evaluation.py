import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer as rs
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from collections import Counter
from nltk import ngrams

fileName = 'Texts/Llama_Sad.json'

# Load generated sentences from JSON file
with open(fileName, 'r') as f:
    data = json.load(f)

# Initialize metric containers
bleu_scores = []
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
perplexities = []
dist1_list, dist2_list, dist3_list = [], [], []

# Set up ROUGE scorer
rouge_scorer = rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Load model and tokenizer
model_name = "Vibuda/llama_trained"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Helper function: Dist-N per sentence
def calc_dist_n(tokens, n):
    n_grams = list(ngrams(tokens, n))
    total = len(n_grams)
    unique = len(set(n_grams))
    return unique / total if total > 0 else 0.0

# Loop over sentences
for sentence in data:
    tokens = sentence.split()

    # BLEU (self-comparison)
    bleu = sentence_bleu([tokens], tokens)
    bleu_scores.append(bleu)

    # ROUGE (self-comparison)
    rouge = rouge_scorer.score(sentence, sentence)
    rouge1_scores.append(rouge['rouge1'].fmeasure)
    rouge2_scores.append(rouge['rouge2'].fmeasure)
    rougeL_scores.append(rouge['rougeL'].fmeasure)

    # Perplexity
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    perplexities.append(perplexity)

    # Dist-1, Dist-2, Dist-3 per sentence
    dist1_list.append(calc_dist_n(tokens, 1))
    dist2_list.append(calc_dist_n(tokens, 2))
    dist3_list.append(calc_dist_n(tokens, 3))

# Function to compute mean and standard error
def mean_and_stderr(lst):
    mean = np.mean(lst)
    stderr = np.std(lst, ddof=1) / np.sqrt(len(lst))
    return mean, stderr

# Compute means and errors
bleu_mean, bleu_err = mean_and_stderr(bleu_scores)
rouge1_mean, rouge1_err = mean_and_stderr(rouge1_scores)
rouge2_mean, rouge2_err = mean_and_stderr(rouge2_scores)
rougeL_mean, rougeL_err = mean_and_stderr(rougeL_scores)
perp_mean, perp_err = mean_and_stderr(perplexities)
dist1_mean, dist1_err = mean_and_stderr(dist1_list)
dist2_mean, dist2_err = mean_and_stderr(dist2_list)
dist3_mean, dist3_err = mean_and_stderr(dist3_list)

# Print final results
print("=== Evaluation Metrics with Standard Errors ===",fileName)
print(f"BLEU: {bleu_mean:.4f} ± {bleu_err:.4f}")
print(f"ROUGE-1: {rouge1_mean:.4f} ± {rouge1_err:.4f}")
print(f"ROUGE-2: {rouge2_mean:.4f} ± {rouge2_err:.4f}")
print(f"ROUGE-L: {rougeL_mean:.4f} ± {rougeL_err:.4f}")
print(f"Perplexity: {perp_mean:.4f} ± {perp_err:.4f}")
print(f"Dist-1 (Unigrams): {dist1_mean:.4f} ± {dist1_err:.4f}")
print(f"Dist-2 (Bigrams): {dist2_mean:.4f} ± {dist2_err:.4f}")
print(f"Dist-3 (Trigrams): {dist3_mean:.4f} ± {dist3_err:.4f}")
