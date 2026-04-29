
print("RICHTIGE DATEI GELADEN")

import pandas as pd

def load_data(path):
    print("Lade Daten von:", path)
    return pd.read_csv(path)