import pandas as pd
import matplotlib.pyplot as plt
import os

# 👉 Pfad zu deiner neuesten Datei anpassen
FILE_PATH = "results/output_2.csv"

df = pd.read_csv(FILE_PATH)

print("=== Überblick ===")
print(df.head())
print("\nAnzahl Zeilen:", len(df))


# -----------------------------
# 📊 Durchschnitt nach temperature
# -----------------------------
temp_group = df.groupby("temperature")["mae"].mean().reset_index()

print("\n=== Durchschnitt MAE nach Temperature ===")
print(temp_group)

# -----------------------------
# 📊 Durchschnitt nach top_p
# -----------------------------
top_p_group = df.groupby("top_p")["mae"].mean().reset_index()

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
print("\nPlot gespeichert: analysis/temperature_vs_mae.png")


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
print("Plot gespeichert: analysis/top_p_vs_mae.png")