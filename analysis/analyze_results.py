import pandas as pd
import matplotlib.pyplot as plt
import os

# 👉 Pfad zu deiner neuesten Datei anpassen
FILE_PATH = "results/output_6.csv"  # 🔁 ggf. anpassen auf neueste Datei


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

# -----------------------------
# 📊 Durchschnitt nach temperature (MAE)
# -----------------------------
temp_group = samples.groupby("temperature")["mae"].mean().reset_index()

print("\n=== Durchschnitt MAE nach Temperature ===")
print(temp_group)

# -----------------------------
# 📊 Durchschnitt nach top_p (MAE)
# -----------------------------
top_p_group = samples.groupby("top_p")["mae"].mean().reset_index()

print("\n=== Durchschnitt MAE nach top_p ===")
print(top_p_group)

# -----------------------------
# 📊 Plot: Temperature vs MAE
# -----------------------------
plt.figure()
plt.plot(temp_group["temperature"], temp_group["mae"], marker='o')
plt.xlabel("Temperature")
plt.ylabel("Mean MAE")
plt.title("Effect of Temperature on MAE")

os.makedirs("analysis", exist_ok=True)
plt.savefig("analysis/temperature_vs_mae.png")
plt.close()

# -----------------------------
# 📊 Plot: top_p vs MAE
# -----------------------------
plt.figure()
plt.plot(top_p_group["top_p"], top_p_group["mae"], marker='o')
plt.xlabel("Top_p")
plt.ylabel("Mean MAE")
plt.title("Effect of Top_p on MAE")

plt.savefig("analysis/top_p_vs_mae.png")
plt.close()

# -----------------------------
# 🔥 NEU: Heatmap (Correlation)
# -----------------------------
# 🔥 Heatmap nur wenn correlation vorhanden
if "correlation" in agg.columns and not agg.empty:

    pivot = agg.pivot_table(
        index="temp_pred" if "temp_pred" in agg.columns else "temperature",
        columns="top_p",
        values="correlation",
        aggfunc="mean"
    )

    plt.figure()
    plt.imshow(pivot, aspect='auto')
    plt.colorbar(label="Correlation")

    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)

    plt.xlabel("top_p")
    plt.ylabel("temp_pred / temperature")
    plt.title("Correlation Heatmap")

    plt.savefig("analysis/heatmap_correlation.png")
    plt.close()
else:
    print("⚠️ Heatmap übersprungen (keine correlation Daten)")

print("\n✅ Analyse abgeschlossen!")
print("📊 Plots gespeichert im analysis/ Ordner")