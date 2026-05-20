import sys
import os


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

from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import os

print(">>> RUN_EXPERIMENT STARTED")


def run():
    df = load_twin2k()
    # TODO: twin2k_loader must provide a 'question' column with held-out questions for proper evaluation
    df = df.sample(samplesize, random_state=42)  # keep for testing

    results = []

    for model_name in MODELS:
        print(f"\n>>> Running model={model_name}")

        for temp_pred in TEMPERATURES:
            print(f"\n>>> Running temp_pred={temp_pred}")
            for reasoning in REASONING_MODES:
                print(f">>> Running reasoning={reasoning}")
                for top_p in [1.0]:  # fixed since focus is temperature + reasoning
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

                        answers = []

                        if reasoning == "cot":
                            reasoning_instruction = "Think step by step before answering."
                        elif reasoning == "uncertain":
                            reasoning_instruction = "Answer like a human with some uncertainty and variability."
                        else:
                            reasoning_instruction = "Answer directly."

                        for question in row["questions"]:
                            prompt = f"""
You are a digital twin of a person with the following personality traits:

Extraversion: {true_traits['Extraversion']}
Agreeableness: {true_traits['Agreeableness']}
Conscientiousness: {true_traits['Conscientiousness']}
Neuroticism: {true_traits['Neuroticism']}
Openness: {true_traits['Openness']}

Instruction:
{reasoning_instruction}

Answer the following question as this person would:

{question}

IMPORTANT:
- Give a natural, realistic answer
- Do not mention traits explicitly
"""

                            # =========================================================
                            # DIGITAL TWIN GENERATION STEP
                            # Independent stateless chat for generating personality answers
                            # =========================================================
                            answer = generate_twin_response(prompt, temp_pred, top_p, model_name)
                            answers.append(answer)

                        # Combine all answers
                        combined_text = "\n".join(answers)

                        # STEP 2: reconstruct traits from ALL answers
                        reconstruction_prompt = f"""
Based on the following answers:

{combined_text}

Estimate Big Five personality traits (values between 0 and 1).

IMPORTANT:
- Output ONLY a single number per trait
- No explanations

Format:
Extraversion: X
Agreeableness: X
Conscientiousness: X
Neuroticism: X
Openness: X
"""

                        # =========================================================
                        # PERSONALITY JUDGE RECONSTRUCTION STEP
                        # Independent stateless chat for reconstructing Big Five traits
                        # from the generated answers only.
                        # =========================================================
                        generated = generate_judge_prediction(reconstruction_prompt, temp_pred, top_p, model_name)
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
                            "input": combined_text,
                            "generated": generated,
                            "mae": score,
                            "type": "sample"
                        })

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

                    corr, _ = pearsonr(flat_pred, flat_true)

                    print(f"✅ Correlation computed: {corr}")

                    print(f"Saving aggregate: temp_pred={temp_pred}, reasoning={reasoning}, top_p={top_p}, corr={corr}, n={len(predictions)}")
                    results.append({
                        "temperature": temp_pred,
                        "model": model_name,
                        "temp_pred": temp_pred,
                        "temp_text": temp_text,
                        "top_p": top_p,
                        "reasoning": reasoning,
                        "correlation": float(corr),
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