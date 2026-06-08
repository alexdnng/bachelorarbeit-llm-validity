import sys
import os
import time


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generation import generate_profile

# =========================================================
# DIGITAL TWIN CHAT
# Separate stateless chat used for personality simulation
# =========================================================
def generate_twin_response(prompt, temperature, top_p, model_name):
    return generate_profile(prompt, temperature, top_p, model_name)


# =========================================================
# PERSONALITY JUDGE CHAT
# Separate stateless chat used ONLY for personality reconstruction
# This ensures conceptual separation between Twin and Judge.
# =========================================================
def generate_judge_prediction(prompt, temperature, top_p, model_name):
    return generate_profile(prompt, temperature, top_p, model_name)


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
    # TODO: twin2k_loader must provide a 'question' column with held-out questions for proper evaluation
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

                            answers = []

                            if reasoning == "think":
                                reasoning_instruction = """
Before answering, consider how a person with the given psychological profile would think, feel, and behave.

Provide only the final answer.
Do not show any reasoning, analysis, or intermediate steps.
"""
                            elif reasoning == "cot":
                                reasoning_instruction = """
Before answering, carefully consider how a person with the given psychological profile would think, feel, and behave.

Think step by step before answering.

Provide only the final answer.
Do not show any reasoning, analysis, or intermediate steps.
"""
                            else:
                                reasoning_instruction = "Answer directly."

                            feature_lines = []

                            for key, value in twin_features.items():
                                label = FEATURE_LABELS[key]
                                feature_lines.append(f"- {label}: {float(value):.2f}")

                            feature_text = "\n".join(feature_lines)

                            for question in row["questions"]:
                                prompt = f"""
You are simulating a real human participant.

The following psychological assessment scores describe this person.
Higher values indicate stronger expression of the characteristic.

Psychological Profile:
{feature_text}

Instruction:
{reasoning_instruction}

Answer the following question as this person would:

{question}

IMPORTANT:
- Give a natural and realistic answer
- Answer consistently with the psychological profile
- Do not mention the profile explicitly
- Only give the answers for the questions
"""

                                # =========================================================
                                # SAFE DIGITAL TWIN GENERATION
                                # Retry mechanism prevents full experiment crashes caused by
                                # temporary API/network/rate-limit errors.
                                # =========================================================
                                answer = None

                                for attempt in range(3):
                                    try:
                                        answer = generate_twin_response(prompt, temp_pred, top_p, model_name)
                                        break
                                    except Exception as e:
                                        print(f"⚠️ Twin generation failed (attempt {attempt+1}/3): {e}")
                                        time.sleep(2)

                                if answer is None:
                                    print("⚠️ Skipping sample because twin generation failed")
                                    continue

                                answers.append(answer)

                            # Combine all answers
                            combined_text = "\n".join(answers)

                            # STEP 2: reconstruct traits from ALL answers
                            reconstruction_prompt = f"""
Based only on the answers below, estimate the person's Big Five personality traits.

Answers:
{combined_text}

Estimate each trait on a continuous scale from 0.00 to 1.00, where higher values indicate stronger expression of the trait.

Use the full range of the scale when justified by the evidence.
Provide precise numerical estimates with two decimal places.
Do not round traits to broad categories such as 0.3, 0.5, or 0.7 unless the evidence genuinely supports those exact values.

Infer the person's relative standing compared to the general population.
Base your estimates only on information contained in the answers.

IMPORTANT:
- Output exactly one numeric value per trait
- Use decimal numbers with four digits after the decimal point
- No explanations
- No additional text

Format:
Extraversion: 0.6325
Agreeableness: 0.7194
Conscientiousness: 0.5493
Neuroticism: 0.8225
Openness: 0.6723
"""

                            # =========================================================
                            # SAFE JUDGE RECONSTRUCTION
                            # Retry mechanism prevents single reconstruction failures
                            # from terminating the entire experiment.
                            # =========================================================
                            generated = None

                            for attempt in range(3):
                                try:
                                    generated = generate_judge_prediction(reconstruction_prompt, 0, top_p, JUDGE_MODEL)
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

                        print(
                            f"Saving aggregate: temp_pred={temp_pred}, reasoning={reasoning}, "
                            f"top_p={top_p}, pearson={pearson_corr}, spearman={spearman_corr}, "
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
                            "pearson_correlation": float(pearson_corr),
                            "spearman_correlation": float(spearman_corr),
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