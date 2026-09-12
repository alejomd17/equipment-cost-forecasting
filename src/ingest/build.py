from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lee X, Y, Z corrigiendo el formato propio de cada archivo.

    X viene en orden descendente y desde 1988; Y usa ';' como separador,
    coma decimal, fecha dd/mm/yyyy y BOM al inicio; Z trae las columnas
    invertidas. Devuelve las tres series por separado, sin recortar, para
    que el pronóstico pueda usar los meses extra de X e Y.
    """
    df_x = pd.read_csv(RAW / "X.csv")
    df_y = pd.read_csv(RAW / "Y.csv", sep=";", decimal=",", encoding="utf-8-sig")
    df_z = pd.read_csv(RAW / "Z.csv")

    for df in [df_x, df_y, df_z]:
        df["Price"] = df["Price"].astype(float)

    df_x = df_x.rename(columns={"Price": "Price_X"})
    df_y = df_y.rename(columns={"Price": "Price_Y"})
    df_z = df_z.rename(columns={"Price": "Price_Z"})

    df_x["Date"] = pd.to_datetime(df_x["Date"], format="%Y-%m-%d")
    df_y["Date"] = pd.to_datetime(df_y["Date"], dayfirst=True)
    df_z["Date"] = pd.to_datetime(df_z["Date"], format="%Y-%m-%d")

    return df_x, df_y, df_z


def merge_raw(df_x: pd.DataFrame, df_y: pd.DataFrame, df_z: pd.DataFrame) -> pd.DataFrame:
    """Une las tres series por fecha.

    Supuesto: solo se conservan los días en que las tres materias primas
    cotizan, que es el criterio con el que se construyó el histórico entregado.
    """
    df = df_x.merge(df_y, on="Date", how="outer").merge(df_z, on="Date", how="outer")
    df = df.dropna().sort_values("Date").reset_index(drop=True)
    return df


def validate_against_historico(df: pd.DataFrame, df_h: pd.DataFrame) -> None:
    """Imprime si fechas y precios de mi unión coinciden con el histórico entregado."""
    fechas_iguales = df["Date"].equals(df_h["Date"])
    print(
        f"{'Fechas iguales' if fechas_iguales else 'Fechas diferentes'} "
        f"- len_df: {len(df)}, len_df_h: {len(df_h)}"
    )

    if not fechas_iguales:
        solo_mio = set(df["Date"]) - set(df_h["Date"])
        solo_hist = set(df_h["Date"]) - set(df["Date"])
        print(len(solo_mio), "solo en mi unión", len(solo_hist), "solo en histórico")

    m = df.merge(df_h[["Date", "Price_X", "Price_Y", "Price_Z"]], on="Date", suffixes=("", "_h"))
    for c in ["Price_X", "Price_Y", "Price_Z"]:
        ok = (m[c] - m[f"{c}_h"]).abs() < 0.01
        print(c, f"{ok.mean():.1%} coinciden")


def main() -> None:
    df = merge_raw(*load_raw())
    df_h = pd.read_csv(RAW / "historico_equipos.csv", parse_dates=["Date"])
    validate_against_historico(df, df_h)

    PROCESSED.mkdir(exist_ok=True)
    df_h.to_csv(PROCESSED / "historico.csv", index=False)


if __name__ == "__main__":
    main()