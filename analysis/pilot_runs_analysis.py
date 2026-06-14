import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg
import os
import argparse

# =========================================================
# ICC STABILITY ANALYSIS
# Pilot study to determine the optimal number of runs.
# Computes ICC for increasing subsets of runs (2, 3, 4, ...)
# to identify the point of convergence.
#
# Usage:
#   python analyze_icc_stability.py --file results/output_1.csv
# =========================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file",
    type=str,
    default="results/output_1.csv",
    help="Path to the pilot results CSV file"
)
args = parser.parse_args()

FILE_PATH = args.file
base_name = os.path.splitext(os.path.basename(FILE_PATH))[0]
os.makedirs("analysis", exist_ok=True)

# -----------------------------
# 1. Datei laden
# -----------------------------
df = pd.read_csv(FILE_PATH)

print(f"✅ Loaded: {FILE_PATH}")
print(f"   Shape: {df.shape}")

# Nur sample rows verwenden
if "type" in df.columns:
    samples = df[df["type"] == "sample"].copy()
else:
    samples = df.copy()

print(f"   Sample rows: {len(samples)}")

# -----------------------------
# 2. Konfiguration
# -----------------------------
trait_cols = [
    "pred_extraversion",
    "pred_agreeableness",
    "pred_conscientiousness",
    "pred_neuroticism",
    "pred_openness",
]

required_cols = ["person_id", "run_id"] + trait_cols

if not all(col in samples.columns for col in required_cols):
    missing = [c for c in required_cols if c not in samples.columns]
    print(f"❌ Missing columns: {missing}")
    exit(1)

max_runs = int(samples["run_id"].max()) + 1
print(f"   Max runs found: {max_runs}")

if max_runs < 2:
    print("❌ At least 2 runs required for ICC stability analysis")
    exit(1)

# -----------------------------
# 3. ICC Stability berechnen
# -----------------------------
print("\n=== Computing ICC Stability ===")

stability_results = []

for n_runs in range(2, max_runs + 1):

    subset = samples[samples["run_id"] < n_runs]

    for trait in trait_cols:

        try:
            icc_subset = subset[["person_id", "run_id", trait]].dropna()

            if icc_subset["person_id"].nunique() < 2:
                continue
            if icc_subset["run_id"].nunique() < 2:
                continue

            icc_table = pg.intraclass_corr(
                data=icc_subset,
                targets="person_id",
                raters="run_id",
                ratings=trait
            )

            # ICC(A,1): absolute agreement, single rater
            icc_row = icc_table.loc[icc_table["Type"] == "ICC(A,1)"]

            # Fallback für ältere Pingouin-Versionen
            if icc_row.empty:
                icc_row = icc_table.loc[icc_table["Type"] == "ICC2"]

            if icc_row.empty:
                print(f"⚠️ No ICC result for n_runs={n_runs}, {trait}")
                continue

            icc_val = float(icc_row["ICC"].iloc[0])
            ci95 = icc_row[["CI95%"]].values[0][0] if "CI95%" in icc_row.columns else [None, None]

            stability_results.append({
                "n_runs": n_runs,
                "trait": trait,
                "icc": icc_val,
                "ci_lower": ci95[0] if ci95 else None,
                "ci_upper": ci95[1] if ci95 else None,
            })

            print(f"   n_runs={n_runs} | {trait}: ICC={icc_val:.3f}")

        except Exception as e:
            print(f"⚠️ ICC failed for n_runs={n_runs}, {trait}: {e}")

# -----------------------------
# 4. Ergebnisse aggregieren
# -----------------------------
stability_df = pd.DataFrame(stability_results)

if stability_df.empty:
    print("❌ No ICC results computed")
    exit(1)

# Mittelwert über alle Traits pro n_runs
mean_stability = (
    stability_df
    .groupby("n_runs")["icc"]
    .mean()
    .reset_index()
    .rename(columns={"icc": "mean_icc"})
)

# Delta ICC: Verbesserung pro zusätzlichem Run
mean_stability["delta_icc"] = mean_stability["mean_icc"].diff().abs()

print("\n=== ICC Stability by Number of Runs ===")
print(mean_stability.to_string(index=False))

# Trait-spezifische Tabelle
trait_stability = (
    stability_df
    .pivot_table(index="n_runs", columns="trait", values="icc")
    .reset_index()
)

print("\n=== ICC per Trait by Number of Runs ===")
print(trait_stability.to_string(index=False))

# -----------------------------
# 5. Speichern
# -----------------------------
mean_stability.to_csv(
    f"analysis/icc_stability_mean_{base_name}.csv",
    index=False
)
stability_df.to_csv(
    f"analysis/icc_stability_full_{base_name}.csv",
    index=False
)
trait_stability.to_csv(
    f"analysis/icc_stability_traits_{base_name}.csv",
    index=False
)

print(f"\n✅ CSVs saved to analysis/")

# -----------------------------
# 6. Plot: Mean ICC by n_runs
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    mean_stability["n_runs"],
    mean_stability["mean_icc"],
    marker='o',
    linewidth=2,
    label="Mean ICC (all traits)"
)

# Benchmarks nach Koo & Li (2016)
plt.axhline(y=0.50, color='red',    linestyle=':', alpha=0.7, label='Poor/Moderate threshold (0.50)')
plt.axhline(y=0.75, color='orange', linestyle='--', alpha=0.7, label='Moderate/Good threshold (0.75)')
plt.axhline(y=0.90, color='green',  linestyle='--', alpha=0.7, label='Good/Excellent threshold (0.90)')

plt.xticks(range(2, max_runs + 1))
plt.xlabel("Number of Runs")
plt.ylabel("Mean ICC")
plt.title("ICC Stability by Number of Runs\n(Pilot Study)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"analysis/icc_stability_mean_{base_name}.png", dpi=150)
plt.close()

# -----------------------------
# 7. Plot: Delta ICC
# -----------------------------
plt.figure(figsize=(8, 4))

delta_data = mean_stability.dropna(subset=["delta_icc"])

plt.bar(
    delta_data["n_runs"],
    delta_data["delta_icc"],
    color="steelblue"
)

# Konvergenzschwelle: Delta < 0.02 gilt als stabil
plt.axhline(y=0.02, color='red', linestyle='--', label='Convergence threshold (ΔICC < 0.02)')

plt.xticks(range(2, max_runs + 1))
plt.xlabel("Number of Runs")
plt.ylabel("ΔICC (improvement per additional run)")
plt.title("Marginal ICC Improvement per Additional Run\n(Pilot Study)")
plt.legend()
plt.tight_layout()
plt.savefig(f"analysis/icc_stability_delta_{base_name}.png", dpi=150)
plt.close()

# -----------------------------
# 8. Plot: ICC per Trait
# -----------------------------
plt.figure(figsize=(10, 5))

for trait in trait_cols:
    trait_data = stability_df[stability_df["trait"] == trait]
    plt.plot(
        trait_data["n_runs"],
        trait_data["icc"],
        marker='o',
        label=trait.replace("pred_", "")
    )

plt.axhline(y=0.75, color='orange', linestyle='--', alpha=0.7, label='Good threshold (0.75)')
plt.axhline(y=0.90, color='green',  linestyle='--', alpha=0.7, label='Excellent threshold (0.90)')

plt.xticks(range(2, max_runs + 1))
plt.xlabel("Number of Runs")
plt.ylabel("ICC")
plt.title("ICC Stability per Trait by Number of Runs\n(Pilot Study)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"analysis/icc_stability_traits_{base_name}.png", dpi=150)
plt.close()

# -----------------------------
# 9. Empfehlung ausgeben
# -----------------------------
print("\n=== Recommendation ===")

convergence_threshold = 0.02
converged_at = None

for _, row in mean_stability.dropna(subset=["delta_icc"]).iterrows():
    if row["delta_icc"] < convergence_threshold:
        converged_at = int(row["n_runs"])
        break

if converged_at:
    icc_at_convergence = mean_stability.loc[
        mean_stability["n_runs"] == converged_at, "mean_icc"
    ].values[0]

    if icc_at_convergence >= 0.90:
        quality = "excellent (Koo & Li, 2016)"
    elif icc_at_convergence >= 0.75:
        quality = "good (Koo & Li, 2016)"
    elif icc_at_convergence >= 0.50:
        quality = "moderate (Koo & Li, 2016)"
    else:
        quality = "poor (Koo & Li, 2016)"

    print(f"✅ ICC converged at n_runs={converged_at} (ΔICC < {convergence_threshold})")
    print(f"   Mean ICC at convergence: {icc_at_convergence:.3f} → {quality}")
    print(f"   Recommendation: Use N_RUNS_PER_SETTING = {converged_at}")
else:
    print(f"⚠️ ICC did not converge within {max_runs} runs (ΔICC always >= {convergence_threshold})")
    print(f"   Consider increasing N_RUNS_PER_SETTING beyond {max_runs}")

print(f"\n📊 Plots saved to analysis/ for {base_name}")
print("✅ ICC Stability Analysis complete!")