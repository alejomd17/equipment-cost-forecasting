from pathlib import Path

import pandas as pd

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

def load_monthly(freq: str = "ME") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga el histórico procesado; devuelve la serie mensual promedio y sus retornos."""
    path = PROCESSED / "historico.csv"
    if not path.exists():
        raise FileNotFoundError("Falta data/processed/historico.csv. Corré 'make data' primero.")

    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    m = df.resample(freq).mean()
    r = m.pct_change().dropna()
    return m, r