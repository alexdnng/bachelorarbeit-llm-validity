"""Assess ICC stability across increasing numbers of repeated inference runs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns


parser = argparse.ArgumentParser(description="Create publication-ready ICC stability tables and figures.")
parser.add_argument("--file", type=str, default="results/output_1.csv", help="Path to the pilot results CSV file.")
args = parser.parse_args()

FILE_PATH = Path(args.file)
BASE_NAME = FILE_PATH.stem
OUTPUT_DIR = Path("analysis") / BASE_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIT_ORDER = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
TRAIT_LABELS = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "neuroticism": "Neuroticism",
}
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]
FIGSIZE = (7.5, 5.0)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_theme(style="whitegrid", rc={"grid.color": "#D9D9D9", "grid.linewidth": 0.7})


def save_table(frame, filename):
    table = frame.copy()
    numeric_columns = table.select_dtypes(include="number").columns
    table[numeric_columns] = table[numeric_columns].round(3)
    table.to_csv(OUTPUT_DIR / filename, index=False, float_format="%.3f")


def save_figure(fig, filename):
    fig.savefig(OUTPUT_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


print(f"Using input file: {FILE_PATH}")
samples = pd.read_csv(FILE_PATH)
samples = samples.loc[samples["type"].eq("sample")].copy() if "type" in samples else samples

trait_columns = [f"pred_{trait}" for trait in TRAIT_ORDER]
required_columns = {"person_id", "run_id", *trait_columns}
missing_columns = required_columns - set(samples.columns)
if missing_columns:
    raise KeyError(f"Pilot analysis is missing required columns: {sorted(missing_columns)}")

run_ids = sorted(samples["run_id"].dropna().unique())
if len(run_ids) < 2:
    raise ValueError("At least two runs are required for ICC stability analysis.")

stability_rows = []
for number_of_runs in range(2, len(run_ids) + 1):
    included_runs = run_ids[:number_of_runs]
    subset = samples.loc[samples["run_id"].isin(included_runs)]
    for trait in TRAIT_ORDER:
        trait_column = f"pred_{trait}"
        icc_data = subset[["person_id", "run_id", trait_column]].dropna()
        if icc_data["person_id"].nunique() < 2 or icc_data["run_id"].nunique() < 2:
            continue
        try:
            icc_table = pg.intraclass_corr(icc_data, targets="person_id", raters="run_id", ratings=trait_column)
            icc_result = icc_table.loc[icc_table["Type"].isin(["ICC(A,1)", "ICC2"])].head(1)
            if icc_result.empty:
                continue
            confidence_interval = icc_result["CI95%"].iloc[0] if "CI95%" in icc_result else [float("nan"), float("nan")]
            stability_rows.append({
                "Number of Runs": number_of_runs,
                "Personality Trait": TRAIT_LABELS[trait],
                "Intraclass Correlation Coefficient Type": "ICC(A,1): Absolute Agreement, Single Measurement",
                "Intraclass Correlation Coefficient (ICC)": float(icc_result["ICC"].iloc[0]),
                "Lower 95% Confidence Interval": confidence_interval[0],
                "Upper 95% Confidence Interval": confidence_interval[1],
            })
        except Exception as error:
            print(f"ICC could not be calculated for {number_of_runs} runs and {trait}: {error}")

stability = pd.DataFrame(stability_rows)
if stability.empty:
    raise ValueError("No ICC results could be calculated from the supplied data.")
stability = stability.sort_values(["Number of Runs", "Personality Trait"], kind="stable")
save_table(stability, "table_icc_stability_by_number_of_runs_and_personality_trait.csv")

mean_stability = stability.groupby("Number of Runs", observed=True)["Intraclass Correlation Coefficient (ICC)"].agg(["mean", "std", "count"]).reset_index()
mean_stability["Absolute Change in Mean ICC from Previous Run Count"] = mean_stability["mean"].diff().abs()
mean_stability = mean_stability.rename(columns={
    "mean": "Mean Intraclass Correlation Coefficient (ICC)",
    "std": "Standard Deviation of ICC Across Personality Traits",
    "count": "Number of Personality Traits",
})
save_table(mean_stability, "table_mean_icc_stability_by_number_of_runs.csv")

trait_stability = stability.pivot(index="Number of Runs", columns="Personality Trait", values="Intraclass Correlation Coefficient (ICC)").reset_index()
trait_stability = trait_stability.reindex(columns=["Number of Runs"] + [TRAIT_LABELS[trait] for trait in TRAIT_ORDER])
save_table(trait_stability, "table_icc_by_number_of_runs_and_personality_trait.csv")

# Mean ICC stability with standard-deviation error bars
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.errorbar(
    mean_stability["Number of Runs"],
    mean_stability["Mean Intraclass Correlation Coefficient (ICC)"],
    yerr=mean_stability["Standard Deviation of ICC Across Personality Traits"].fillna(0),
    marker="o", markersize=6, capsize=4, linewidth=1.6, color="#0072B2", ecolor="#333333",
    label="Mean across Big Five personality traits (± 1 standard deviation)",
)
for _, row in mean_stability.iterrows():
    ax.annotate(f"{row['Mean Intraclass Correlation Coefficient (ICC)']:.3f}", (row["Number of Runs"], row["Mean Intraclass Correlation Coefficient (ICC)"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=10)
ax.axhline(0.50, color="#666666", linestyle=":", linewidth=1, label="ICC reference threshold: 0.50")
ax.axhline(0.75, color="#666666", linestyle="--", linewidth=1, label="ICC reference threshold: 0.75")
ax.axhline(0.90, color="#333333", linestyle="-.", linewidth=1, label="ICC reference threshold: 0.90")
ax.set_xlabel("Number of Repeated Inference Runs")
ax.set_ylabel("Mean Intraclass Correlation Coefficient (ICC)")
ax.set_xticks(mean_stability["Number of Runs"])
ax.set_title("")
ax.legend(frameon=False, loc="lower right")
ax.margins(y=0.12)
save_figure(fig, "figure_mean_icc_stability_by_number_of_runs")

# Marginal ICC change
delta_data = mean_stability.dropna(subset=["Absolute Change in Mean ICC from Previous Run Count"])
fig, ax = plt.subplots(figsize=FIGSIZE)
bars = ax.bar(delta_data["Number of Runs"], delta_data["Absolute Change in Mean ICC from Previous Run Count"], color="#0072B2", edgecolor="#333333", linewidth=0.6)
for bar, value in zip(bars, delta_data["Absolute Change in Mean ICC from Previous Run Count"]):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
ax.axhline(0.02, color="#333333", linestyle="--", linewidth=1, label="Convergence threshold: absolute ICC change < 0.02")
ax.set_xlabel("Number of Repeated Inference Runs")
ax.set_ylabel("Absolute Change in Mean Intraclass Correlation Coefficient (ICC)")
ax.set_xticks(mean_stability["Number of Runs"])
ax.set_title("")
ax.legend(frameon=False)
ax.margins(y=0.16)
save_figure(fig, "figure_marginal_change_in_mean_icc_by_number_of_runs")

# Trait-specific ICC stability
fig, ax = plt.subplots(figsize=FIGSIZE)
for color, trait in zip(PALETTE, [TRAIT_LABELS[item] for item in TRAIT_ORDER]):
    trait_data = stability.loc[stability["Personality Trait"] == trait]
    ax.plot(trait_data["Number of Runs"], trait_data["Intraclass Correlation Coefficient (ICC)"], marker="o", markersize=5, linewidth=1.5, color=color, label=trait)
ax.axhline(0.75, color="#666666", linestyle="--", linewidth=1, label="ICC reference threshold: 0.75")
ax.axhline(0.90, color="#333333", linestyle="-.", linewidth=1, label="ICC reference threshold: 0.90")
ax.set_xlabel("Number of Repeated Inference Runs")
ax.set_ylabel("Intraclass Correlation Coefficient (ICC)")
ax.set_xticks(mean_stability["Number of Runs"])
ax.set_title("")
ax.legend(title="Big Five Personality Trait", frameon=False, loc="lower right")
ax.margins(y=0.10)
save_figure(fig, "figure_icc_stability_by_personality_trait")

# Retain the existing convergence recommendation, now based on the formatted summary table.
converged = mean_stability.loc[mean_stability["Absolute Change in Mean ICC from Previous Run Count"] < 0.02]
if not converged.empty:
    recommendation = int(converged.iloc[0]["Number of Runs"])
    print(f"ICC convergence criterion first met at {recommendation} repeated inference runs.")
else:
    print("The ICC convergence criterion was not met within the available repeated inference runs.")
print("Pilot ICC stability analysis completed successfully.")
