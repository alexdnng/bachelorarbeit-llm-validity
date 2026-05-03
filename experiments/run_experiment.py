from src.generation import generate_profile, generate_behavior_from_traits
from src.evaluation import parse_traits, compute_mae
from src.config import *
from src.twin2k_loader import load_twin2k

from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import os

print(">>> RUN_EXPERIMENT STARTED")


def run():
    df = load_twin2k()
    df = df.sample(20, random_state=42)  # 🔥 limit for faster testing

    results = []

    for temp_pred in TEMPERATURES:
        print(f"\n>>> Running temp_pred={temp_pred}")
        for top_p in TOP_P_VALUES:
            print(f">>> Running top_p={top_p}")

            temp_text = 0.8  # fixed for text generation

            predictions = []
            ground_truth = []

            for i, (_, row) in enumerate(df.iterrows()):
                print(f"Processing sample {i+1}/{len(df)}")

                true_traits = {
                    "Extraversion": row["Extraversion"],
                    "Agreeableness": row["Agreeableness"],
                    "Conscientiousness": row["Conscientiousness"],
                    "Neuroticism": row["Neuroticism"],
                    "Openness": row["Openness"]
                }

                # STEP 1: generate behavior text
                behavior_text = generate_behavior_from_traits(true_traits, temp_text, top_p)

                # STEP 2: predict traits from behavior
                prompt = f"""
Based on the following behavior:

{behavior_text}

Estimate Big Five personality traits (values between 0 and 1).

IMPORTANT:
- Output ONLY a single number per trait
- Do NOT include ranges, ±, explanations, or text
- Use decimal values between 0 and 1

Format:
Extraversion: X
Agreeableness: X
Conscientiousness: X
Neuroticism: X
Openness: X
"""

                generated = generate_profile(prompt, temp_pred, top_p)
                pred_traits = parse_traits(generated)

                # 🔥 FIX: ungültige Predictions rausfiltern
                if len(pred_traits) != 5:
                    continue

                if any(v is None for v in pred_traits.values()):
                    continue

                pred_values = list(pred_traits.values())
                true_values = list(true_traits.values())

                predictions.append(pred_values)
                ground_truth.append(true_values)

                score = compute_mae(pred_traits, true_traits)

                results.append({
                    "temperature": temp_pred,
                    "temp_pred": temp_pred,
                    "temp_text": temp_text,
                    "top_p": top_p,
                    "input": behavior_text,
                    "generated": generated,
                    "mae": score,
                    "type": "sample"
                })

            # 🔥 Safety checks before correlation
            if len(predictions) < 5:
                print("⚠️ Too few valid samples – skipping correlation")
                continue

            flat_pred = np.array(predictions).flatten()
            flat_true = np.array(ground_truth).flatten()

            if len(set(flat_pred)) < 2:
                print("⚠️ No variance in predictions – skipping correlation")
                continue

            corr, _ = pearsonr(flat_pred, flat_true)

            results.append({
                "temperature": temp_pred,
                "temp_pred": temp_pred,
                "temp_text": temp_text,
                "top_p": top_p,
                "correlation": corr,
                "n_samples": len(predictions),
                "type": "aggregate"
            })

    return results


def get_next_filename(folder="results", base="output"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    existing_files = os.listdir(folder)
    numbers = []

    for file in existing_files:
        if file.startswith(base) and file.endswith(".csv"):
            try:
                num = int(file.replace(base + "_", "").replace(".csv", ""))
                numbers.append(num)
            except:
                pass

    next_number = max(numbers) + 1 if numbers else 1
    return f"{folder}/{base}_{next_number}.csv"


if __name__ == "__main__":
    results = run()
    df = pd.DataFrame(results)

    filename = get_next_filename()
    df.to_csv(filename, index=False)

    print(f"Gespeichert als: {filename}")
    print("Experiment fertig!")