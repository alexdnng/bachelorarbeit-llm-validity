import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import re



# ============================================================
# Automatically find newest output_X.csv
# ============================================================

results_dir = "results"

output_files = []

for filename in os.listdir(results_dir):
    match = re.match(r"output_(\d+)\.csv$", filename)
    if match:
        output_files.append((int(match.group(1)), filename))

if not output_files:
    raise FileNotFoundError(
        "No output_X.csv file found in results directory"
    )

latest_index, latest_file = max(
    output_files,
    key=lambda x: x[0]
)

FILE_PATH = Path(results_dir) / latest_file
BASE_NAME = Path(latest_file).stem

print(f"📂 Using file: {FILE_PATH}")

OUTPUT_DIR = Path("analysis") / BASE_NAME / "inference_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Saving results to: {OUTPUT_DIR}")


# ============================================================
# Helper Functions
# ============================================================


def run_anova(df, dependent_var):
    model = ols(
        f"{dependent_var} ~ C(temperature) * C(reasoning)",
        data=df
    ).fit()

    anova = sm.stats.anova_lm(
        model,
        typ=2
    )

    total_ss = anova["sum_sq"].sum()

    anova["eta_sq"] = anova["sum_sq"] / total_ss

    residual_ss = anova.loc["Residual", "sum_sq"]

    anova["partial_eta_sq"] = (
        anova["sum_sq"] /
        (anova["sum_sq"] + residual_ss)
    )

    return model, anova



def print_results(title, table):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(
        table[
            ["df", "F", "PR(>F)", "eta_sq", "partial_eta_sq"]
        ].round(4)
    )



def interpret_effect(p):
    if pd.isna(p):
        return "-"
    if p < 0.001:
        return "highly significant"
    if p < 0.05:
        return "significant"
    return "not significant"



def print_interpretation(title, table):
    print(f"\n{title} interpretation:")

    for effect in table.index:
        if effect == "Residual":
            continue

        p = table.loc[effect, "PR(>F)"]
        eta = table.loc[effect, "eta_sq"]

        print(
            f"- {effect}: {interpret_effect(p)} "
            f"(p={p:.4f}, eta²={eta:.3f})"
        )



def check_normality(model, name):
    stat, p = shapiro(model.resid)

    print(f"\n{name} residual normality")
    print(f"Shapiro-Wilk statistic = {stat:.4f}")
    print(f"Shapiro-Wilk p-value  = {p:.4f}")



def save_interaction_plot(df, dependent_var, filename, ylabel):
    plt.figure(figsize=(8, 5))

    sns.pointplot(
        data=df,
        x="temperature",
        y=dependent_var,
        hue="reasoning"
    )

    plt.ylabel(ylabel)
    plt.xlabel("Temperature")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / filename)
    plt.close()



def run_tukey_if_significant(df, variable, factor, anova_table):
    effect_name = f"C({factor})"

    if effect_name not in anova_table.index:
        return

    p = anova_table.loc[effect_name, "PR(>F)"]

    if p >= 0.05:
        return

    print(f"\nTukey HSD for {variable} ~ {factor}")

    tukey = pairwise_tukeyhsd(
        endog=df[variable],
        groups=df[factor],
        alpha=0.05
    )

    print(tukey)


# ============================================================
# Load Data
# ============================================================

print("Loading data...")

df = pd.read_csv(FILE_PATH)

agg = df[df["type"] == "aggregate"].copy()

print(f"Aggregate rows: {len(agg)}")


# ============================================================
# Descriptive Statistics
# ============================================================

# Pearson / Spearman aus den Aggregate-Zeilen
desc_corr = agg.groupby(
    ["temperature", "reasoning"]
)[[
    "pearson_mean",
    "spearman_mean"
]].agg(["mean", "std"])

# MAE aus den Sample-Zeilen berechnen
sample_desc = df[df["type"] == "sample"].copy()

desc_mae = (
    sample_desc
    .groupby(
        ["temperature", "reasoning"]
    )["mae"]
    .agg(["mean", "std"])
)

# gleiche MultiIndex-Struktur wie desc_corr erzeugen
desc_mae.columns = pd.MultiIndex.from_product(
    [["mae"], desc_mae.columns]
)


# Zusammenführen
desc = desc_corr.join(desc_mae)

desc.to_csv(
    OUTPUT_DIR / "descriptive_stats.csv"
)

# ============================================================
# Pearson
# ============================================================

model_pearson, anova_pearson = run_anova(
    agg,
    "pearson_mean"
)

print_results(
    "PEARSON MEAN ANOVA",
    anova_pearson
)

print_interpretation(
    "Pearson",
    anova_pearson
)

check_normality(
    model_pearson,
    "Pearson"
)

save_interaction_plot(
    agg,
    "pearson_mean",
    "interaction_pearson.png",
    "Pearson Mean"
)


# ============================================================
# Spearman
# ============================================================

model_spearman, anova_spearman = run_anova(
    agg,
    "spearman_mean"
)

print_results(
    "SPEARMAN MEAN ANOVA",
    anova_spearman
)

print_interpretation(
    "Spearman",
    anova_spearman
)

check_normality(
    model_spearman,
    "Spearman"
)

save_interaction_plot(
    agg,
    "spearman_mean",
    "interaction_spearman.png",
    "Spearman Mean"
)


# ============================================================
# MAE
# ============================================================

sample_df = df[df["type"] == "sample"].copy()

mae_agg = (
    sample_df
    .groupby(
        ["temperature", "reasoning", "run"]
    )["mae"]
    .mean()
    .reset_index()
)

model_mae, anova_mae = run_anova(
    mae_agg,
    "mae"
)

print_results(
    "MAE ANOVA",
    anova_mae
)

print_interpretation(
    "MAE",
    anova_mae
)

check_normality(
    model_mae,
    "MAE"
)

save_interaction_plot(
    mae_agg,
    "mae",
    "interaction_mae.png",
    "MAE"
)


# ============================================================
# Optional Tukey Tests
# ============================================================

run_tukey_if_significant(
    agg,
    "pearson_mean",
    "temperature",
    anova_pearson
)

run_tukey_if_significant(
    agg,
    "pearson_mean",
    "reasoning",
    anova_pearson
)

run_tukey_if_significant(
    agg,
    "spearman_mean",
    "temperature",
    anova_spearman
)

run_tukey_if_significant(
    agg,
    "spearman_mean",
    "reasoning",
    anova_spearman
)

run_tukey_if_significant(
    mae_agg,
    "mae",
    "temperature",
    anova_mae
)

run_tukey_if_significant(
    mae_agg,
    "mae",
    "reasoning",
    anova_mae
)


# ============================================================
# Save Results
# ============================================================

anova_pearson.to_csv(
    OUTPUT_DIR / "anova_pearson.csv"
)

anova_spearman.to_csv(
    OUTPUT_DIR / "anova_spearman.csv"
)

anova_mae.to_csv(
    OUTPUT_DIR / "anova_mae.csv"
)

print("\nSaved files:")
print(f"- {OUTPUT_DIR}/anova_pearson.csv")
print(f"- {OUTPUT_DIR}/anova_spearman.csv")
print(f"- {OUTPUT_DIR}/anova_mae.csv")
print(f"- {OUTPUT_DIR}/descriptive_stats.csv")
print(f"- {OUTPUT_DIR}/interaction_pearson.png")
print(f"- {OUTPUT_DIR}/interaction_spearman.png")
print(f"- {OUTPUT_DIR}/interaction_mae.png")
print(f"\n✅ Inference analysis completed for {BASE_NAME}")