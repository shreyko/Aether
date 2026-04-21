"""BLEU, token-level F1, and LLM-as-a-judge scoring for RAG baseline."""

import json
import re
from typing import Dict

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from .config import DEFAULT_MAX_TOKENS, VLLM_MODEL, get_vllm_client
from .prompts import ACCURACY_PROMPT

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Reuse a single OpenAI-compatible client across all judge calls. The openai
# client is thread-safe and pools HTTP connections, so this avoids rebuilding
# the client and its socket pool on every scored item.
_JUDGE_CLIENT = get_vllm_client()


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
    prompt = ACCURACY_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )
    response = _JUDGE_CLIENT.chat.completions.create(
        model=VLLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    text = ""
    try:
        text = response.choices[0].message.content or ""
    except Exception:
        text = response.choices[0]["message"]["content"] if response.choices else ""

    json_match = re.search(r"\{[^}]*\}", text)
    if json_match:
        try:
            label = json.loads(json_match.group()).get("label", "")
            return 1 if label == "CORRECT" else 0
        except (json.JSONDecodeError, KeyError):
            pass

    upper = text.upper()
    correct_pos = upper.rfind("CORRECT")
    wrong_pos = upper.rfind("WRONG")
    if correct_pos > wrong_pos:
        return 1
    return 0
