import pandas as pd

from src.config import (
    QUESTIONS,
    TWIN_FEATURE_COLUMNS,
    GROUND_TRUTH_COLUMNS,
)

def load_twin2k():
    """
    Loads Twin-2K data and prepares it for the experiment.

    Key decisions:
    - Big Five traits are used ONLY as ground truth
    - Digital twins receive multidimensional psychological constructs
    - Multiple waves are merged via TWIN_ID
    - Participants with missing feature values are removed
    """

    # Load waves
    wave1 = pd.read_csv("data/mega_persona_summary_csv/wave 1 scores.csv")
    wave2 = pd.read_csv("data/mega_persona_summary_csv/wave 2 scores.csv")
    wave3 = pd.read_csv("data/mega_persona_summary_csv/wave 3 scores.csv")

    # Remove duplicate ground-truth columns from later waves
    wave2 = wave2.drop(columns=GROUND_TRUTH_COLUMNS, errors="ignore")
    wave3 = wave3.drop(columns=GROUND_TRUTH_COLUMNS, errors="ignore")

    # Merge waves
    df = wave1.merge(wave2, on="TWIN_ID")
    df = df.merge(wave3, on="TWIN_ID")



    # Remove participants with missing feature values
    df = df.dropna(subset=TWIN_FEATURE_COLUMNS)

    processed = []

    for _, row in df.iterrows():

        twin_features = {
            col: row[col]
            for col in TWIN_FEATURE_COLUMNS
        }

        ground_truth = {
            "Extraversion":      (row["score_extraversion"] - 1) / 4,
            "Agreeableness":     (row["score_agreeableness"] - 1) / 4,
            "Conscientiousness": (row["score_conscientiousness"] - 1) / 4,
            "Neuroticism":       (row["score_neuroticism"] - 1) / 4,
            "Openness":          (row["score_openness"] - 1) / 4,
}

        processed.append({
            "id": row["TWIN_ID"],
            "twin_features": twin_features,
            "ground_truth": ground_truth,
            "questions": QUESTIONS
        })

    return pd.DataFrame(processed)