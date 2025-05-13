"""Utility functions for computing metrics."""

import torch
import numpy as np
from typing import Dict, Any, List
from collections import Counter

def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    return np.exp(loss)

def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor, pad_token_id: int = 0) -> float:
    """Compute token-level accuracy, ignoring padding tokens."""
    mask = targets != pad_token_id
    correct = (predictions.argmax(dim=-1)[mask] == targets[mask]).float().sum()
    total = mask.float().sum()
    return (correct / total).item()

def compute_bleu(predictions: List[str], references: List[str], n_gram: int = 4) -> float:
    """Compute BLEU score."""
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] += 1
        return ngrams
    
    def compute_bleu_score(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        pred_ngrams = get_ngrams(pred_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        
        if not pred_ngrams:
            return 0.0
            
        matches = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())
        
        return matches / total if total > 0 else 0.0
    
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        # Compute scores for different n-gram sizes
        n_scores = []
        for n in range(1, n_gram + 1):
            score = compute_bleu_score(pred_tokens, ref_tokens, n)
            n_scores.append(score)
            
        # Geometric mean of n-gram scores
        if any(score == 0 for score in n_scores):
            scores.append(0.0)
        else:
            score = np.exp(np.mean([np.log(s) for s in n_scores]))
            scores.append(score)
            
    return np.mean(scores)

def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss: float,
    pad_token_id: int = 0
) -> Dict[str, float]:
    """Compute all metrics for the model."""
    metrics = {
        'loss': loss,
        'perplexity': compute_perplexity(loss),
        'accuracy': compute_accuracy(predictions, targets, pad_token_id)
    }
    
    return metrics 