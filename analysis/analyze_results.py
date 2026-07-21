import pandas as pd
import os
import re
import pingouin as pg

import seaborn as sns

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# =========================================================
# Automatisch neueste output_X.csv finden
# =========================================================
# =========================================================
# Datei auswählen
# =========================================================

USE_LATEST_OUTPUT = False
MANUAL_FILE = "output_9.csv"

results_dir = "results"

if USE_LATEST_OUTPUT:

    output_files = []

    for filename in os.listdir(results_dir):
        match = re.match(r"output_(\d+)\.csv$", filename)
        if match:
            output_files.append((int(match.group(1)), filename))

    if not output_files:
        raise FileNotFoundError("Keine output_X.csv Datei im results-Ordner gefunden")

    _, selected_file = max(output_files, key=lambda x: x[0])

else:

    selected_file = MANUAL_FILE

FILE_PATH = os.path.join(results_dir, selected_file)
base_name = os.path.splitext(selected_file)[0]


print(f"📂 Verwende Datei: {FILE_PATH}")

# =========================================================
# Analyseordner für diesen Output erstellen
# =========================================================
analysis_output_dir = os.path.join("analysis", base_name)
os.makedirs(analysis_output_dir, exist_ok=True)

print(f"📁 Ergebnisse werden gespeichert in: {analysis_output_dir}")


# 🔧 1. Datei laden
df = pd.read_csv(FILE_PATH)


print("COLUMNS:", df.columns)
print("SHAPE:", df.shape)

if "pearson_mean" in df.columns:
    print("Pearson Mean Werte:")
    print(df["pearson_mean"].describe())

if "spearman_mean" in df.columns:
    print("Spearman Mean Werte:")
    print(df["spearman_mean"].describe())

print("=== Überblick ===")
print(df.head())
print("\nAnzahl Zeilen:", len(df))

# 🔧 2. Daten trennen (robust gegen fehlende "type"-Spalte)
if "type" in df.columns:
    agg = df.loc[df["type"] == "aggregate"]
    samples = df.loc[df["type"] == "sample"]
else:
    print("⚠️ 'type' Spalte nicht gefunden – fallback wird verwendet")

    # gesamte Daten als samples behandeln
    samples = df.copy()

    # aggregate selbst berechnen (nur wenn Pearson existiert)
    if "pearson_mean" in df.columns:
        agg = df[df["pearson_mean"].notna()].copy()
    else:
        agg = pd.DataFrame(columns=df.columns)

# 🔧 Fallback: Falls keine Pearson correlation existiert
if "pearson_mean" not in df.columns:
    print("⚠️ Keine 'pearson_mean' Spalte gefunden – Heatmap wird übersprungen")


print("Anzahl agg rows:", len(agg))
print("AGG COLUMNS:", agg.columns)
print("AGG DTYPES:\n", agg.dtypes)
print(agg.head())

# -----------------------------
# 📊 Per-Trait MAE Analysis
# -----------------------------
trait_mapping = {
    "extraversion": ("pred_extraversion", "gt_extraversion"),
    "agreeableness": ("pred_agreeableness", "gt_agreeableness"),
    "conscientiousness": ("pred_conscientiousness", "gt_conscientiousness"),
    "neuroticism": ("pred_neuroticism", "gt_neuroticism"),
    "openness": ("pred_openness", "gt_openness"),
}

# If your ground-truth columns use different names, adjust here:
# For example, if your CSV uses "gt_extraversion", change "true_extraversion" to "gt_extraversion" etc.

available_traits = {}

for trait_name, (pred_col, true_col) in trait_mapping.items():
    if pred_col in df.columns and true_col in df.columns:
        available_traits[trait_name] = (pred_col, true_col)

if available_traits:

    trait_mae_rows = []

    for trait_name, (pred_col, true_col) in available_traits.items():

        tmp = samples[[pred_col, true_col]].dropna().copy()
        tmp["trait_mae"] = (tmp[pred_col] - tmp[true_col]).abs()

        trait_mae_rows.append({
            "trait": trait_name,
            "mean_mae": tmp["trait_mae"].mean(),
            "std_mae": tmp["trait_mae"].std(),
        })

    trait_mae_df = pd.DataFrame(trait_mae_rows)

    print("\n=== Per-Trait MAE ===")
    print(trait_mae_df)

    trait_mae_df.to_csv(
        os.path.join(
            analysis_output_dir,
            f"per_trait_mae_{base_name}.csv"
        ),
        index=False,
        float_format="%.3f"
    )
    trait_mae_df.to_excel(
        os.path.join(
            analysis_output_dir,
            f"per_trait_mae_{base_name}.xlsx"
        ),
        index=False,
        float_format="%.3f"
    )
    

    plt.figure(figsize=(8, 4))
    plt.bar(
        trait_mae_df["trait"],
        trait_mae_df["mean_mae"],
        yerr=trait_mae_df["std_mae"],
    )
    plt.ylabel("Mean MAE")
    plt.xlabel("Trait")
    #plt.title("Per-Trait Prediction Error")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            analysis_output_dir,
            f"per_trait_mae_{base_name}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    
    plt.savefig(
    os.path.join(
        analysis_output_dir,
        f"per_trait_mae_{base_name}.pdf"
    ),
    bbox_inches="tight"
)
    plt.close()

else:
    print("⚠️ Per-trait MAE skipped: ground-truth trait columns not found")

# -----------------------------
# 📊 Durchschnitt nach temperature (MAE)
# -----------------------------
temp_group = (
    samples
    .groupby("temperature")["mae"]
    .agg(["mean", "std"])
    .reset_index()
)

print("\n=== Durchschnitt MAE nach Temperature ===")
print(temp_group[["temperature", "mean", "std"]])

# -----------------------------
# 📊 Durchschnitt nach reasoning (MAE)
# -----------------------------
if "reasoning" in samples.columns:
    reasoning_group = (
        samples
        .groupby("reasoning")["mae"]
        .agg(["mean", "std"])
        .reset_index()
    )

    print("\n=== Durchschnitt MAE nach Reasoning ===")
    print(reasoning_group[["reasoning", "mean", "std"]])
else:
    print("⚠️ 'reasoning' Spalte nicht gefunden – überspringe Reasoning-Analyse")

# -----------------------------
# 📊 Plot: Temperature vs MAE
# -----------------------------
plt.figure(figsize=(8, 5))
plt.errorbar(
    temp_group["temperature"],
    temp_group["mean"],
    yerr=temp_group["std"],
    marker='o'
)
plt.xlabel("Temperature")
plt.ylabel("Mean MAE")
#plt.title("Effect of Temperature on MAE")

plt.tight_layout()

plt.savefig(
    os.path.join(
        analysis_output_dir,
        f"temperature_vs_mae_{base_name}.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    os.path.join(
        analysis_output_dir,
        f"temperature_vs_mae_{base_name}.pdf"
    ),
    bbox_inches="tight"
)
plt.close()

# -----------------------------
# 📊 Plot: Reasoning vs MAE
# -----------------------------
if "reasoning" in samples.columns:
    plt.figure(figsize=(8, 5))
    plt.bar(
        reasoning_group["reasoning"],
        reasoning_group["mean"],
        yerr=reasoning_group["std"]
    )
    plt.xlabel("Reasoning Mode")
    plt.ylabel("Mean MAE")
    #plt.title("Effect of Reasoning on MAE")
    
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            analysis_output_dir,
            f"reasoning_vs_mae_{base_name}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    
    plt.savefig(
    os.path.join(
        analysis_output_dir,
        f"reasoning_vs_mae_{base_name}.pdf"
    ),
    bbox_inches="tight"
)
    plt.close()

# -----------------------------
# 🔥 Heatmaps (Pearson + Spearman)
# -----------------------------
# 🔥 Heatmaps nur wenn Correlation-Daten vorhanden
if "pearson_mean" in agg.columns and not agg.empty:
    # Ensure numeric types for pivot axes
    for col in ["temp_pred", "temperature", "pearson_mean", "spearman_mean"]:
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors="coerce")

    # Drop rows with missing correlation or axes
    pivot_index = "temp_pred" if "temp_pred" in agg.columns else "temperature"
    if "reasoning" in agg.columns:
        agg_clean = agg.dropna(subset=[pivot_index, "reasoning", "pearson_mean", "spearman_mean"]).copy()
    else:
        print("⚠️ 'reasoning' fehlt für Heatmap")
        agg_clean = pd.DataFrame()

    print("Cleaned agg size:", len(agg_clean))

    # Optionally: quick sanity assertion
    assert not agg_clean.empty, "Keine gültigen Daten für Heatmap nach Cleaning"

    # =========================================================
    # PEARSON HEATMAP
    # =========================================================
    pearson_pivot = agg_clean.pivot_table(
        index=pivot_index,
        columns="reasoning",
        values="pearson_mean",
        aggfunc="mean"
    )

    pearson_pivot = pearson_pivot.sort_index().sort_index(axis=1)

    print("PEARSON PIVOT TABLE:\n", pearson_pivot)

    if pearson_pivot.empty or pearson_pivot.isna().all().all():
        print("⚠️ Pearson Pivot leer oder nur NaN – keine Heatmap möglich")
    else:
        plt.figure(figsize=(7, 5))

        sns.heatmap(
            pearson_pivot,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar_kws={"label": "Pearson correlation"}
        )

        plt.xlabel("Reasoning")
        plt.ylabel("Temperature")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                analysis_output_dir,
                f"heatmap_pearson_{base_name}.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.savefig(
            os.path.join(
                analysis_output_dir,
                f"heatmap_pearson_{base_name}.pdf"
            ),
            bbox_inches="tight"
        )

        plt.close()
        

    # =========================================================
    # SPEARMAN HEATMAP
    # =========================================================
    spearman_pivot = agg_clean.pivot_table(
        index=pivot_index,
        columns="reasoning",
        values="spearman_mean",
        aggfunc="mean"
    )

    spearman_pivot = spearman_pivot.sort_index().sort_index(axis=1)

    print("SPEARMAN PIVOT TABLE:\n", spearman_pivot)

    if spearman_pivot.empty or spearman_pivot.isna().all().all():
        print("⚠️ Spearman Pivot leer oder nur NaN – keine Heatmap möglich")
    else:
        plt.figure(figsize=(7, 5))

        sns.heatmap(
            spearman_pivot,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar_kws={"label": "Spearman correlation"}
        )

        plt.xlabel("Reasoning")
        plt.ylabel("Temperature")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                analysis_output_dir,
                f"heatmap_spearman_{base_name}.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.savefig(
            os.path.join(
                analysis_output_dir,
                f"heatmap_spearman_{base_name}.pdf"
            ),
            bbox_inches="tight"
        )

        plt.close()
        
else:
    print("⚠️ Heatmaps übersprungen (keine Correlation-Daten)")

 # -----------------------------
# 📈 Variabilitätsanalyse der Traits
# -----------------------------
trait_cols = [
    "pred_extraversion",
    "pred_agreeableness",
    "pred_conscientiousness",
    "pred_neuroticism",
    "pred_openness",
]

required_cols = ["person_id", "temperature", "reasoning"] + trait_cols

if all(col in samples.columns for col in required_cols):
    variability = (
        samples
        .groupby(["person_id", "temperature", "reasoning"])[trait_cols]
        .std()
        .reset_index()
    )

    variability["mean_trait_std"] = variability[trait_cols].mean(axis=1)

    print("\n=== Variability by Person ===")
    print(variability.head())

    variability.to_csv(
        os.path.join(
            analysis_output_dir,
            f"trait_variability_{base_name}.csv"
        ),
        index=False,
        float_format="%.3f"
    )
    variability.to_excel(
        os.path.join(
            analysis_output_dir,
            f"trait_variability_{base_name}.xlsx"
        ),
        index=False,
        float_format="%.3f"
    )

    temp_variability = (
        variability
        .groupby("temperature")["mean_trait_std"]
        .agg(["mean", "std"])
        .reset_index()
    )

    print("\n=== Mean Trait STD by Temperature ===")
    print(temp_variability)

    reasoning_variability = (
        variability
        .groupby("reasoning")["mean_trait_std"]
        .agg(["mean", "std"])
        .reset_index()
    )

    print("\n=== Mean Trait STD by Reasoning ===")
    print(reasoning_variability)
else:
    print("⚠️ Variability analysis skipped: required trait columns missing")

# -----------------------------
# 📈 ICC Analyse
# -----------------------------
if "run_id" in samples.columns:

    icc_results = []

    for temperature in sorted(samples["temperature"].unique()):
        for reasoning in sorted(samples["reasoning"].unique()):

            subset = samples[
                (samples["temperature"] == temperature)
                & (samples["reasoning"] == reasoning)
            ]

            for trait in trait_cols:

                try:

                    icc_subset = subset[
                        ["person_id", "run_id", trait]
                    ].dropna()

                    # ICC benötigt mindestens zwei Personen
                    # und mindestens zwei Rater (Runs)
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

                    # Neuere Pingouin-Versionen verwenden
                    # ICC(1,1), ICC(A,1), ICC(C,1) statt ICC1/ICC2/ICC3.
                    icc_row = icc_table.loc[
                        icc_table["Type"] == "ICC(A,1)"
                    ]

                    # Fallback für ältere Pingouin-Versionen
                    if icc_row.empty:
                        icc_row = icc_table.loc[
                            icc_table["Type"] == "ICC2"
                        ]

                    if icc_row.empty:
                        print(
                            f"⚠️ No suitable ICC result for "
                            f"{temperature}, {reasoning}, {trait}"
                        )
                        continue

                    icc2 = float(icc_row["ICC"].iloc[0])

                    icc_results.append({
                        "temperature": temperature,
                        "reasoning": reasoning,
                        "trait": trait,
                        "icc": icc2
                    })

                except Exception as e:
                    print(
                        f"⚠️ ICC failed for "
                        f"{temperature}, {reasoning}, {trait}: {e}"
                    )

    print(f"\nCollected ICC results: {len(icc_results)}")
    icc_df = pd.DataFrame(icc_results)

    if not icc_df.empty:

        print("\n=== ICC Results ===")
        print(icc_df.head())
        
        icc_export = icc_df.copy()
            
        icc_export["icc"] = icc_export["icc"].round(3)
        

        icc_export.to_csv(
            os.path.join(
                analysis_output_dir,
                f"icc_results_{base_name}.csv"
            ),
            index=False
        )
        icc_export.to_excel(
            os.path.join(
                analysis_output_dir,
                f"icc_results_{base_name}.xlsx"
            ),
            index=False,
        )

        mean_icc = (
            icc_df
            .groupby(["temperature", "reasoning"])["icc"]
            .mean()
            .reset_index()
        )

        print("\n=== Mean ICC by Condition ===")
        print(mean_icc)

        mean_icc.to_csv(
            os.path.join(
                analysis_output_dir,
                f"mean_icc_{base_name}.csv"
            ),
            index=False,
            float_format="%.3f"
        )
        mean_icc.to_excel(
            os.path.join(
                analysis_output_dir,
                f"mean_icc_{base_name}.xlsx"
            ),
            index=False,
            float_format="%.3f"
        )
        

else:
    print("⚠️ run_id missing - ICC skipped")

print("\n✅ Analyse abgeschlossen!")
print(f"📊 Ergebnisse gespeichert in: {analysis_output_dir}")