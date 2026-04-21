"""
BLEU, token-level F1, and LLM-as-a-judge scoring.

BLEU / F1 utilities adapted from
https://github.com/mem0ai/mem0/blob/main/evaluation/metrics/utils.py
(originally from AgenticMemory).

LLM judge adapted from
https://github.com/mem0ai/mem0/blob/main/evaluation/metrics/llm_judge.py
"""

import json
import re
from typing import Dict

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from .config import VLLM_MODEL, get_vllm_client
from .prompts import ACCURACY_PROMPT

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)


def _simple_tokenize(text: str) -> list[str]:
    return str(text).lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]
    smooth = SmoothingFunction().method1

    weights_list = [
        (1, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (0.33, 0.33, 0.33, 0),
        (0.25, 0.25, 0.25, 0.25),
    ]
    scores: Dict[str, float] = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            scores[f"bleu{n}"] = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        except Exception:
            scores[f"bleu{n}"] = 0.0
    return scores


def calculate_f1(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return 0.0

    pred_tokens = set(_simple_tokenize(prediction))
    ref_tokens = set(_simple_tokenize(reference))
    common = pred_tokens & ref_tokens

    if not pred_tokens or not ref_tokens:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_llm_judge(question: str, gold_answer: str, generated_answer: str) -> int:
    """Return 1 if the LLM judge deems the answer CORRECT, else 0."""
    client = get_vllm_client()
    prompt = ACCURACY_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )
    response = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = response.choices[0].message.content or ""

    # Try JSON extraction first
    json_match = re.search(r"\{[^}]*\}", text)
    if json_match:
        try:
            label = json.loads(json_match.group())["label"]
            return 1 if label == "CORRECT" else 0
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: look for the last occurrence of CORRECT / WRONG
    upper = text.upper()
    correct_pos = upper.rfind("CORRECT")
    wrong_pos = upper.rfind("WRONG")
    if correct_pos > wrong_pos:
        return 1
    return 0
