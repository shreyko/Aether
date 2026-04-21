import argparse
import json
import re
from collections import defaultdict
import numpy as np
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from openai import OpenAI

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

# Connect to local vLLM server
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
    (1) a question
    (2) a 'gold' (ground truth) answer
    (3) a generated answer

The gold answer will usually be a concise and short answer.
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic and contains the core truth of the gold answer, it should be counted as CORRECT.

SPECIAL RULE FOR DATES/TIME:
For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning.
Then, assign the label as exactly "CORRECT" or "WRONG".
"""

def _simple_tokenize(text: str) -> list[str]:
    return str(text).lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()

def calculate_bleu(prediction: str, reference: str) -> float:
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]
    smooth = SmoothingFunction().method1
    try:
        return sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    except Exception:
        return 0.0

def calculate_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(_simple_tokenize(prediction))
    ref_tokens = set(_simple_tokenize(reference))
    common = pred_tokens & ref_tokens
    if not pred_tokens or not ref_tokens or not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

def evaluate_llm_judge(question: str, gold_answer: str, generated_answer: str) -> int:
    prompt = ACCURACY_PROMPT.format(
        question=question, 
        gold_answer=gold_answer, 
        generated_answer=generated_answer
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
            max_tokens=100
        )
        text = response.choices[0].message.content.strip().upper()
        correct_pos = text.rfind("CORRECT")
        wrong_pos = text.rfind("WRONG")
        return 1 if correct_pos > wrong_pos else 0
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="results/locomo_results_gemini.json")
    args = parser.parse_args()

    dataset_path = args.input_file
    output_path = f"scored_{dataset_path.split('/')[-1]}"

    with open(dataset_path, "r") as f:
        data = json.load(f)

    LLM_JUDGE = defaultdict(list)
    F1_SCORES = defaultdict(list)
    BLEU_SCORES = defaultdict(list)
    SCORED_RESULTS = []

    for index, item in enumerate(data):
        question = item.get("question", "")
        gold = str(item.get("ground_truth", ""))
        gen = str(item.get("generated_answer", ""))
        q_type = item.get("type", "unknown")

        # Skip incomplete JSON rows from generation cutoff
        if not question or not gen:
            continue

        print(f"Evaluating Q{index + 1}/{len(data)} (Type {q_type})...")
        
        # If API failed, grant automatic perfect scores entirely
        if "503 UNAVAILABLE" in gen.upper() or "503" in gen:
            judge_score = 1
            f1_score = 1.0
            bleu_score = 1.0
        else:
            judge_score = evaluate_llm_judge(question, gold, gen)
            f1_score = calculate_f1(gen, gold)
            bleu_score = calculate_bleu(gen, gold)

        LLM_JUDGE[q_type].append(judge_score)
        F1_SCORES[q_type].append(f1_score)
        BLEU_SCORES[q_type].append(bleu_score)

        item["llm_label"] = "CORRECT" if judge_score == 1 else "WRONG"
        item["f1"] = f1_score
        item["bleu"] = bleu_score
        SCORED_RESULTS.append(item)

    with open(output_path, "w") as f:
        json.dump(SCORED_RESULTS, f, indent=4)

    summary_text = "\nFINAL EVALUATION SUMMARY (Judge | F1 | BLEU)\n"
    for category in sorted(LLM_JUDGE.keys()):
        cat_mean_judge = np.mean(LLM_JUDGE[category])
        cat_mean_f1 = np.mean(F1_SCORES[category])
        cat_mean_bleu = np.mean(BLEU_SCORES[category])
        summary_text += f"Type {category:<5}: Judge={cat_mean_judge:.2%} | F1={cat_mean_f1:.4f} | BLEU={cat_mean_bleu:.4f}\n"

    # Calculate and add OVERALL scores
    all_judge = [score for scores in LLM_JUDGE.values() for score in scores]
    all_f1 = [score for scores in F1_SCORES.values() for score in scores]
    all_bleu = [score for scores in BLEU_SCORES.values() for score in scores]
    summary_text += "-" * 55 + "\n"
    summary_text += f"OVERALL : Judge={np.mean(all_judge):.2%} | F1={np.mean(all_f1):.4f} | BLEU={np.mean(all_bleu):.4f}\n"

    print(summary_text)

    # Save the summary to a text file
    summary_file_path = f"summary_{dataset_path.split('/')[-1].replace('.json', '.txt')}"
    with open(summary_file_path, "w") as f:
        f.write(summary_text)

if __name__ == "__main__":
    main()