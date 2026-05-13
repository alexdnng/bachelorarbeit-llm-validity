import pandas as pd
import matplotlib.pyplot as plt
import os

# 👉 Pfad zu deiner neuesten Datei anpassen
FILE_PATH = "results/output_9.csv"  # 🔁 ggf. anpassen auf neueste Datei
# 🔥 Dateiname dynamisch extrahieren
base_name = os.path.splitext(os.path.basename(FILE_PATH))[0]


# 🔧 1. Datei laden
df = pd.read_csv(FILE_PATH)


print("COLUMNS:", df.columns)
print("SHAPE:", df.shape)

if "correlation" in df.columns:
    print("Correlation Werte:", df["correlation"].describe())
else:
    print("❌ KEINE correlation Spalte vorhanden")

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

    # aggregate selbst berechnen (nur wenn correlation existiert)
    if "correlation" in df.columns:
        agg = df[df["correlation"].notna()].copy()
    else:
        agg = pd.DataFrame(columns=df.columns)

# 🔧 Fallback: Falls keine correlation existiert → selbst berechnen (optional)
if "correlation" not in df.columns:
    print("⚠️ Keine 'correlation' Spalte gefunden – Heatmap wird übersprungen")

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
# 🔥 NEU: Heatmap (Correlation)
# -----------------------------
# 🔥 Heatmap nur wenn correlation vorhanden
if "correlation" in agg.columns and not agg.empty:
    # Ensure numeric types for pivot axes
    for col in ["temp_pred", "temperature", "correlation"]:
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors="coerce")

    # Drop rows with missing correlation or axes
    pivot_index = "temp_pred" if "temp_pred" in agg.columns else "temperature"
    if "reasoning" in agg.columns:
        agg_clean = agg.dropna(subset=[pivot_index, "reasoning", "correlation"]).copy()
    else:
        print("⚠️ 'reasoning' fehlt für Heatmap")
        agg_clean = pd.DataFrame()

    print("Cleaned agg size:", len(agg_clean))

    # Optionally: quick sanity assertion
    assert not agg_clean.empty, "Keine gültigen Daten für Heatmap nach Cleaning"

    pivot = agg_clean.pivot_table(
        index=pivot_index,
        columns="reasoning",
        values="correlation",
        aggfunc="mean"
    )

    # Sort axes for consistent plotting
    pivot = pivot.sort_index().sort_index(axis=1)

    print("PIVOT TABLE:\n", pivot)

    if pivot.empty or pivot.isna().all().all():
        print("⚠️ Pivot leer oder nur NaN – keine Heatmap möglich")
    else:
        plt.figure()

        # Replace NaN with a sentinel for visibility
        data = pivot.fillna(0).values

        im = plt.imshow(data, aspect='auto')
        plt.colorbar(im, label="Correlation")

        plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
        plt.yticks(range(len(pivot.index)), [round(i, 3) for i in pivot.index])

        plt.xlabel("Reasoning Mode")
        plt.ylabel(pivot_index)
        plt.title("Correlation Heatmap (Temperature × Reasoning)")

        plt.tight_layout()
        plt.savefig(f"analysis/heatmap_correlation_{base_name}.png")
        plt.close()
else:
    print("⚠️ Heatmap übersprungen (keine correlation Daten)")

print("\n✅ Analyse abgeschlossen!")
print(f"📊 Plots gespeichert im analysis/ Ordner für {base_name}")