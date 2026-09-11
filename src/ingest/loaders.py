"""Loaders para los archivos crudos de materias primas y equipos.

Cada loader devuelve un DataFrame con columnas Date (datetime64) y Price (float),
ordenado ascendente por fecha, sin fechas duplicadas.
"""

from pathlib import Path

import pandas as pd

COLUMNS = ["Date", "Price"]


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el formato común a cualquier loader."""
    df = df[COLUMNS].copy()
    df["Price"] = df["Price"].astype(float)
    df = df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    return df


def load_y(path: Path) -> pd.DataFrame:
    """Y.csv usa ';' como separador, coma decimal, fecha dd/mm/yyyy y BOM al inicio."""
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return _standardize(df)


def load_x(path: Path) -> pd.DataFrame:
    """X.csv viene en orden descendente y arranca en 1988."""
    df = pd.read_csv(path, parse_dates=["Date"])
    return _standardize(df)


def load_z(path: Path) -> pd.DataFrame:
    """Z.csv tiene las columnas invertidas (Price, Date)."""
    ...


def load_historico(path: Path) -> pd.DataFrame:
    """historico_equipos.csv ya viene limpio; devuelve las 6 columnas tal cual."""
    ...