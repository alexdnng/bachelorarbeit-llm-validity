import pandas as pd
import matplotlib.pyplot as plt
import os

# 👉 Pfad zu deiner neuesten Datei anpassen
FILE_PATH = "results/output_1.csv"  # 🔁 ggf. anpassen auf neueste Datei
# 🔥 Dateiname dynamisch extrahieren
base_name = os.path.splitext(os.path.basename(FILE_PATH))[0]


# 🔧 1. Datei laden
df = pd.read_csv(FILE_PATH)


print("COLUMNS:", df.columns)
print("SHAPE:", df.shape)

if "pearson_correlation" in df.columns:
    print("Pearson Werte:")
    print(df["pearson_correlation"].describe())

if "spearman_correlation" in df.columns:
    print("Spearman Werte:")
    print(df["spearman_correlation"].describe())

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
    if "pearson_correlation" in df.columns:
        agg = df[df["pearson_correlation"].notna()].copy()
    else:
        agg = pd.DataFrame(columns=df.columns)

# 🔧 Fallback: Falls keine Pearson correlation existiert
if "pearson_correlation" not in df.columns:
    print("⚠️ Keine 'pearson_correlation' Spalte gefunden – Heatmap wird übersprungen")

print("Anzahl agg rows:", len(agg))
print("AGG COLUMNS:", agg.columns)
print("AGG DTYPES:\n", agg.dtypes)
print(agg.head())

# -----------------------------
# 📊 Durchschnitt nach temperature (MAE)
# -----------------------------
temp_group = samples.groupby("temperature")["mae"].mean().reset_index()

print("\n=== Durchschnitt MAE nach Temperature ===")
print(temp_group)

# -----------------------------
# 📊 Durchschnitt nach reasoning (MAE)
# -----------------------------
if "reasoning" in samples.columns:
    reasoning_group = samples.groupby("reasoning")["mae"].mean().reset_index()

    print("\n=== Durchschnitt MAE nach Reasoning ===")
    print(reasoning_group)
else:
    print("⚠️ 'reasoning' Spalte nicht gefunden – überspringe Reasoning-Analyse")

# -----------------------------
# 📊 Plot: Temperature vs MAE
# -----------------------------
plt.figure()
plt.plot(temp_group["temperature"], temp_group["mae"], marker='o')
plt.xlabel("Temperature")
plt.ylabel("Mean MAE")
plt.title("Effect of Temperature on MAE")

os.makedirs("analysis", exist_ok=True)
plt.savefig(f"analysis/temperature_vs_mae_{base_name}.png")
plt.close()

# -----------------------------
# 📊 Plot: Reasoning vs MAE
# -----------------------------
if "reasoning" in samples.columns:
    plt.figure()
    plt.bar(reasoning_group["reasoning"], reasoning_group["mae"])
    plt.xlabel("Reasoning Mode")
    plt.ylabel("Mean MAE")
    plt.title("Effect of Reasoning on MAE")

    plt.savefig(f"analysis/reasoning_vs_mae_{base_name}.png")
    plt.close()

# -----------------------------
# 🔥 Heatmaps (Pearson + Spearman)
# -----------------------------
# 🔥 Heatmaps nur wenn Correlation-Daten vorhanden
if "pearson_correlation" in agg.columns and not agg.empty:
    # Ensure numeric types for pivot axes
    for col in ["temp_pred", "temperature", "pearson_correlation", "spearman_correlation"]:
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors="coerce")

    # Drop rows with missing correlation or axes
    pivot_index = "temp_pred" if "temp_pred" in agg.columns else "temperature"
    if "reasoning" in agg.columns:
        agg_clean = agg.dropna(subset=[pivot_index, "reasoning", "pearson_correlation", "spearman_correlation"]).copy()
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
        values="pearson_correlation",
        aggfunc="mean"
    )

    pearson_pivot = pearson_pivot.sort_index().sort_index(axis=1)

    print("PEARSON PIVOT TABLE:\n", pearson_pivot)

    if pearson_pivot.empty or pearson_pivot.isna().all().all():
        print("⚠️ Pearson Pivot leer oder nur NaN – keine Heatmap möglich")
    else:
        plt.figure()

        data = pearson_pivot.fillna(0).values

        im = plt.imshow(data, aspect='auto')
        plt.colorbar(im, label="Pearson Correlation")

        plt.xticks(range(len(pearson_pivot.columns)), [str(c) for c in pearson_pivot.columns])
        plt.yticks(range(len(pearson_pivot.index)), [round(i, 3) for i in pearson_pivot.index])

        plt.xlabel("Reasoning Mode")
        plt.ylabel(pivot_index)
        plt.title("Pearson Correlation Heatmap")

        plt.tight_layout()
        plt.savefig(f"analysis/heatmap_pearson_{base_name}.png")
        plt.close()

    # =========================================================
    # SPEARMAN HEATMAP
    # =========================================================
    spearman_pivot = agg_clean.pivot_table(
        index=pivot_index,
        columns="reasoning",
        values="spearman_correlation",
        aggfunc="mean"
    )

    spearman_pivot = spearman_pivot.sort_index().sort_index(axis=1)

    print("SPEARMAN PIVOT TABLE:\n", spearman_pivot)

    if spearman_pivot.empty or spearman_pivot.isna().all().all():
        print("⚠️ Spearman Pivot leer oder nur NaN – keine Heatmap möglich")
    else:
        plt.figure()

        data = spearman_pivot.fillna(0).values

        im = plt.imshow(data, aspect='auto')
        plt.colorbar(im, label="Spearman Correlation")

        plt.xticks(range(len(spearman_pivot.columns)), [str(c) for c in spearman_pivot.columns])
        plt.yticks(range(len(spearman_pivot.index)), [round(i, 3) for i in spearman_pivot.index])

        plt.xlabel("Reasoning Mode")
        plt.ylabel(pivot_index)
        plt.title("Spearman Correlation Heatmap")

        plt.tight_layout()
        plt.savefig(f"analysis/heatmap_spearman_{base_name}.png")
        plt.close()
else:
    print("⚠️ Heatmaps übersprungen (keine Correlation-Daten)")

print("\n✅ Analyse abgeschlossen!")
print(f"📊 Plots gespeichert im analysis/ Ordner für {base_name}")