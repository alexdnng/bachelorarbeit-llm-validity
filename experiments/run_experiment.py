from src.data_loader import load_data
from src.generation import generate_profile
from src.evaluation import parse_traits, compute_mae
from src.config import *

import pandas as pd
import os

print("Script Startet")


def run():
    df = load_data("data/test_data.csv")

    results = []

    for temp in TEMPERATURES:
        for top_p in TOP_P_VALUES:
            for _, row in df.iterrows():

                prompt = f"""
Based on the following behavior:

{row['input']}

Estimate Big Five personality traits (values between 0 and 1).

There may be uncertainty in the personality estimation.
Reflect this uncertainty in your scores.

Format:
Extraversion: X
Agreeableness: X
Conscientiousness: X
Neuroticism: X
Openness: X
"""

                # 👉 API CALL
                generated = generate_profile(prompt, temp, top_p)

                # 👉 Debug (kannst du später entfernen)
                print("PROMPT:", row["input"])
                print("GENERATED:", generated)

                pred_traits = parse_traits(generated)

                print("PARSED:", pred_traits)

                true_traits = {
                    "Extraversion": row["Extraversion"],
                    "Agreeableness": row["Agreeableness"],
                    "Conscientiousness": row["Conscientiousness"],
                    "Neuroticism": row["Neuroticism"],
                    "Openness": row["Openness"]
                }

                score = compute_mae(pred_traits, true_traits)

                print("MAE:", score)
                print("------")

                results.append({
                    "temperature": temp,
                    "top_p": top_p,
                    "input": row["input"],
                    "generated": generated,
                    "mae": score
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