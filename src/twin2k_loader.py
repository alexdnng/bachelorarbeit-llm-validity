import pandas as pd

from src.config import QUESTIONS

def load_twin2k():
    """
    Loads Twin-2K Wave 1 data and prepares it for the experiment.

    Key decisions:
    - Only Wave 1 is used (consistent Big Five measurement)
    - Big Five traits are normalized to [0,1]
    - A fixed set of standardized questions is attached to each person
    """

    # Load ONLY Wave 1 (cleanest personality ground truth)
    df = pd.read_csv("data/mega_persona_summary_csv/wave 1 scores.csv")

    # Fixed question set (same for every participant → ensures comparability)
    

    processed = []

    for _, row in df.iterrows():
        processed.append({
            "id": row["TWIN_ID"],

            # Big Five (normalized to [0,1])
            "Extraversion": row["score_extraversion"] / 5,
            "Agreeableness": row["score_agreeableness"] / 5,
            "Conscientiousness": row["score_conscientiousness"] / 5,
            "Neuroticism": row["score_neuroticism"] / 5,
            "Openness": row["score_openness"] / 5,

            # Same questions for every person (important for experimental control)
            "questions": QUESTIONS
        })

    return pd.DataFrame(processed)