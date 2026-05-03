import pandas as pd


def load_twin2k():
    df1 = pd.read_csv("data/mega_persona_summary_csv/wave 1 scores.csv")
    df2 = pd.read_csv("data/mega_persona_summary_csv/wave 2 scores.csv")
    df3 = pd.read_csv("data/mega_persona_summary_csv/wave 3 scores.csv")

    df = pd.concat([df1, df2, df3], ignore_index=True)

    processed = []

    for _, row in df.iterrows():
        processed.append({
            "Extraversion": row["score_extraversion"] / 5,
            "Agreeableness": row["score_agreeableness"] / 5,
            "Conscientiousness": row["score_conscientiousness"] / 5,
            "Neuroticism": row["score_neuroticism"] / 5,
            "Openness": row["score_openness"] / 5
        })

    return pd.DataFrame(processed)