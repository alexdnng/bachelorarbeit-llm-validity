import src.evaluation as ev
print(dir(ev))
from src.data_loader import load_data
from src.generation import generate_profile
from src.evaluation import parse_traits, compute_mae
from src.config import *
from src.evaluation import parse_traits, compute_mae

print("Script Startet")

def run():
    df = load_data("data/test_data.csv")

    results = []
    for _ in range(N_RUNS_PER_SETTING):
        for temp in TEMPERATURES:
            for top_p in TOP_P_VALUES:
                for _, row in df.iterrows():

                    prompt = f"""
                    Based on the following behavior:

                    {row['input']}

                    Estimate Big Five personality traits.
                    Return ONLY numbers between 0 and 1.

                    Format:
                    Extraversion: X
                    Agreeableness: X
                    Conscientiousness: X
                    Neuroticism: X
                    Openness: X
                    """

                    generated = generate_profile(prompt, temp, top_p)

                    pred_traits = parse_traits(generated)

                    true_traits = {

                        "Extraversion": row["Extraversion"],

                        "Agreeableness": row["Agreeableness"],

                        "Conscientiousness": row["Conscientiousness"],

                        "Neuroticism": row["Neuroticism"],

                        "Openness": row["Openness"]

                    }

                    score = compute_mae(pred_traits, true_traits)

                    results.append({
                        "temperature": temp,
                        "top_p": top_p,
                        "input": row['input'],
                        "generated": generated,
                        "score": score
                    })

        return results


if __name__ == "__main__":
    results = run()

    import pandas as pd
    df = pd.DataFrame(results)
    import os

    def get_next_filename(folder="results", base="output"):
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


    filename = get_next_filename()
    df.to_csv(filename, index=False)

    print(f"Gespeichert als: {filename}")

    print("Experiment fertig!")