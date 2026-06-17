import sys
import os
import time


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generation import (
    generate_profile,
    generate_twin_response,
    generate_judge_response,
)



from src.evaluation import parse_traits, compute_mae
from src.config import *
from src.twin2k_loader import load_twin2k


from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
import os

print(">>> RUN_EXPERIMENT STARTED")

# =========================================================
# INCREMENTAL RESULT SAVING
# Temporary checkpoint file used during execution to avoid
# data loss in case of crashes or API failures.
# =========================================================
SAVE_FILE = "results/temp_results.csv"


def run():
    df = load_twin2k()
    df = df.sample(samplesize, random_state=42)  # keep for testing

    results = []

    # Create results folder if necessary
    os.makedirs("results", exist_ok=True)

    for model_name in MODELS:
        print(f"\n>>> Running model={model_name}")

        for temp_pred in TEMPERATURES:
            print(f"\n>>> Running temp_pred={temp_pred}")
            for reasoning in REASONING_MODES:
                print(f">>> Running reasoning={reasoning}")
                for top_p in [1.0]:  # fixed since focus is temperature + reasoning
                    print(f">>> Running top_p={top_p}")

                    for run_idx in range(N_RUNS_PER_SETTING):
                        print(f">>> Run {run_idx + 1}/{N_RUNS_PER_SETTING}")

                        temp_text = 0  # fixed for text generation

                        predictions = []
                        ground_truth = []

                        for i, (_, row) in enumerate(df.iterrows()):
                            print(f"Processing sample {i+1}/{len(df)}")

                            true_traits = row["ground_truth"]
                            twin_features = row["twin_features"]

                            if reasoning == "think":
                                reasoning_instruction = """
Before answering, form a mental model of the person described by the psychological profile.

Consider how this person would typically think, feel, and behave in everyday life.

Answer the questions from this person's perspective.

Provide only the final answers.
Do not show any reasoning.
"""
                            elif reasoning == "cot":
                                reasoning_instruction = """
Before answering, carefully analyze the psychological profile.

Reason step by step about how the traits combine to form a coherent personality.

Form a detailed mental model of the person and answer the questions from that person's perspective.

Provide only the final answers.
Do not show any reasoning.
"""
                            else:
                                reasoning_instruction = """
Answer each question immediately as the person would.

Do not perform explicit analysis.
Provide only the answers.
"""

                            feature_lines = []

                            for key, value in twin_features.items():
                                label = FEATURE_LABELS[key]
                                feature_lines.append(f"- {label}: {float(value):.2f}")

                            feature_text = "\n".join(feature_lines)

                            questions_text = "\n\n".join(row["questions"])

                            prompt = f"""
You are simulating a real human participant.

The following psychological assessment scores describe this person.
Higher values indicate stronger expression of the characteristic.

Psychological Profile:
{feature_text}

Instruction:
{reasoning_instruction}

Answer all of the following questions as this person would.

{questions_text}

IMPORTANT:
- Give natural and realistic answers
- Answer consistently with the psychological profile
- Do not mention the profile explicitly
- Provide one answer for each question
- Preserve the question numbering in your response
"""

                            combined_text = None

                            for attempt in range(3):
                                try:
                                    combined_text = generate_twin_response(
                                        prompt,
                                        temp_pred,
                                        top_p,
                                        model_name,
                                    )
                                    break
                                except Exception as e:
                                    print(f"⚠️ Twin generation failed (attempt {attempt+1}/3): {e}")
                                    time.sleep(2)

                            if combined_text is None:
                                print("⚠️ Skipping sample because twin generation failed")
                                continue

                            # STEP 2: reconstruct traits from ALL answers
                            reconstruction_prompt = f"""
Based on the questions and answers below, estimate the person's Big Five personality traits.

Questions:
{questions_text}

Answers:
{combined_text.strip()}

Estimate each trait on a continuous scale from 0.00 to 1.00, where higher values indicate stronger expression of the trait.

Base your estimates only on the behavioral evidence contained in the answers.

IMPORTANT:
- Output exactly one numeric value per trait
- No explanations
- No additional text

Format:
Extraversion: <value>
Agreeableness: <value>
Conscientiousness: <value>
Neuroticism: <value>
Openness: <value>
"""

                            # =========================================================
                            # SAFE JUDGE RECONSTRUCTION
                            # Retry mechanism prevents single reconstruction failures
                            # from terminating the entire experiment.
                            # =========================================================
                            generated = None

                            for attempt in range(3):
                                try:
                                    generated = generate_judge_response(
                                        reconstruction_prompt,
                                        0,
                                        top_p,
                                        JUDGE_MODEL,
                                    )
                                    break
                                except Exception as e:
                                    print(f"⚠️ Judge reconstruction failed (attempt {attempt+1}/3): {e}")
                                    time.sleep(2)

                            if generated is None:
                                print("⚠️ Skipping sample because judge reconstruction failed")
                                continue

                            pred_traits = parse_traits(generated)

                            # 🔥 FIX: ungültige Predictions rausfiltern
                            if len(pred_traits) != 5:
                                continue

                            if any(v is None for v in pred_traits.values()):
                                continue

                            pred_values = list(pred_traits.values())
                            true_values = list(true_traits.values())

                            # Ensure numeric conversion
                            pred_values = [float(v) for v in pred_values]
                            true_values = [float(v) for v in true_values]

                            predictions.append(pred_values)
                            ground_truth.append(true_values)

                            score = compute_mae(pred_traits, true_traits)

                            results.append({
                                "temperature": temp_pred,
                                "model": model_name,
                                "temp_pred": temp_pred,
                                "temp_text": temp_text,
                                "top_p": top_p,
                                "reasoning": reasoning,
                                "run": run_idx,
                                "run_id": run_idx,
                                "person_id": row["id"],
                                "samplesize": samplesize,
                                "input": combined_text,
                                "generated": generated,
                                "pred_traits": str(pred_traits),
                                "pred_extraversion": pred_traits["Extraversion"],
                                "pred_agreeableness": pred_traits["Agreeableness"],
                                "pred_conscientiousness": pred_traits["Conscientiousness"],
                                "pred_neuroticism": pred_traits["Neuroticism"],
                                "pred_openness": pred_traits["Openness"],
                                "true_traits": str(true_traits),
                                "gt_extraversion": true_traits["Extraversion"],
                                "gt_agreeableness": true_traits["Agreeableness"],
                                "gt_conscientiousness": true_traits["Conscientiousness"],
                                "gt_neuroticism": true_traits["Neuroticism"],
                                "gt_openness": true_traits["Openness"],
                                "mae": score,
                                "type": "sample"
                            })

                            # =========================================================
                            # INCREMENTAL CHECKPOINT SAVE
                            # Save intermediate results after every processed sample.
                            # =========================================================
                            pd.DataFrame(results).to_csv(SAVE_FILE, index=False)

                        # 🔥 Safety checks before correlation
                        if len(predictions) < 5:
                            print("⚠️ Too few valid samples – skipping correlation")
                            continue

                        # Convert to numpy arrays
                        pred_array = np.array(predictions, dtype=float)
                        true_array = np.array(ground_truth, dtype=float)

                        trait_names = [
                            "Extraversion",
                            "Agreeableness",
                            "Conscientiousness",
                            "Neuroticism",
                            "Openness",
                        ]

                        trait_correlations = {}

                        for i, trait in enumerate(trait_names):
                            pred_trait = pred_array[:, i]
                            true_trait = true_array[:, i]

                            if np.std(pred_trait) == 0 or np.std(true_trait) == 0:
                                trait_correlations[trait] = np.nan
                            else:
                                r, _ = pearsonr(pred_trait, true_trait)
                                trait_correlations[trait] = float(r)

                        trait_spearman = {}

                        for i, trait in enumerate(trait_names):
                            pred_trait = pred_array[:, i]
                            true_trait = true_array[:, i]

                            if np.std(pred_trait) == 0 or np.std(true_trait) == 0:
                                trait_spearman[trait] = np.nan
                            else:
                                r, _ = spearmanr(pred_trait, true_trait)
                                trait_spearman[trait] = float(r)

                        mean_trait_pearson = float(
                            np.nanmean(list(trait_correlations.values()))
                        )
                        mean_trait_spearman = float(
                            np.nanmean(list(trait_spearman.values()))
                        )

                        flat_pred = np.array(predictions, dtype=float).flatten()
                        flat_true = np.array(ground_truth, dtype=float).flatten()

                        # Remove NaNs explicitly
                        mask = ~np.isnan(flat_pred) & ~np.isnan(flat_true)
                        flat_pred = flat_pred[mask]
                        flat_true = flat_true[mask]

                        # Final safety checks
                        if len(flat_pred) < 2:
                            print("⚠️ Not enough valid values after cleaning – skipping correlation")
                            continue

                        if np.std(flat_pred) == 0 or np.std(flat_true) == 0:
                            print("⚠️ No variance after cleaning – skipping correlation")
                            continue

                        # =========================================================
                        # CORRELATION ANALYSIS
                        # Pearson measures linear agreement between predicted and
                        # true trait values.
                        # Spearman measures rank-order agreement and is additionally
                        # appropriate for ordinal / Likert-style personality data.
                        # =========================================================
                        pearson_corr, _ = pearsonr(flat_pred, flat_true)
                        spearman_corr, _ = spearmanr(flat_pred, flat_true)

                        print(f"✅ Pearson correlation computed: {pearson_corr}")
                        print(f"✅ Spearman correlation computed: {spearman_corr}")
                        print("✅ Trait-specific Pearson correlations:")
                        for trait, value in trait_correlations.items():
                            print(f"   {trait}: {value}")
                        print(f"✅ Mean trait Pearson correlation: {mean_trait_pearson}")
                        print(f"✅ Mean trait Spearman correlation: {mean_trait_spearman}")

                        print(
                            f"Saving aggregate: temp_pred={temp_pred}, reasoning={reasoning}, "
                            f"top_p={top_p}, pearson_mean={mean_trait_pearson}, global_flat={pearson_corr}, spearman_mean={mean_trait_spearman}, spearman_flat={spearman_corr}, "
                            f"n={len(predictions)}"
                        )

                        results.append({
                            "temperature": temp_pred,
                            "model": model_name,
                            "temp_pred": temp_pred,
                            "temp_text": temp_text,
                            "top_p": top_p,
                            "reasoning": reasoning,
                            "run": run_idx,
                            "samplesize": samplesize,
                            "pearson_global_flat": float(pearson_corr),
                            "pearson_mean": mean_trait_pearson,
                            "pearson_extraversion": trait_correlations["Extraversion"],
                            "pearson_agreeableness": trait_correlations["Agreeableness"],
                            "pearson_conscientiousness": trait_correlations["Conscientiousness"],
                            "pearson_neuroticism": trait_correlations["Neuroticism"],
                            "pearson_openness": trait_correlations["Openness"],
                            "spearman_global_flat": float(spearman_corr),
                            "spearman_mean": mean_trait_spearman,
                            "spearman_extraversion": trait_spearman["Extraversion"],
                            "spearman_agreeableness": trait_spearman["Agreeableness"],
                            "spearman_conscientiousness": trait_spearman["Conscientiousness"],
                            "spearman_neuroticism": trait_spearman["Neuroticism"],
                            "spearman_openness": trait_spearman["Openness"],
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