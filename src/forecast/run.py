"""Pronóstico final de materias primas y equipos, con intervalos por bootstrap

La relación entre equipos y materias primas es determinística (ver 02_modeling),
por lo que toda la incertidumbre del costo de los equipos proviene del pronóstico
de X, Y y Z. Los intervalos se construyen remuestreando los errores que cada
modelo cometió en el backtest, paso a paso del horizonte
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.forecast.commodities import MODELOS

ROOT = Path(__file__).resolve().parents[2]
FORECASTS = ROOT / "data" / "forecasts"

# Modelo ganador por serie según el backtest multi-ventana (ver 03_forecast)
import json

SELECCION_PATH = ROOT / "models" / "seleccion.json"
if not SELECCION_PATH.exists():
    raise FileNotFoundError("Falta models/seleccion.json. Corré 'make train' primero")
SELECCION = json.loads(SELECCION_PATH.read_text())

GANADORES = json.loads(SELECCION_PATH.read_text())["ganadores"]

# Fórmulas validadas en 02_modeling
FORMULAS = {
    "Equipo1": {"X": 0.2, "Y": 0.8, "Z": 0.0},
    "Equipo2": {"X": 1 / 3, "Y": 1 / 3, "Z": 1 / 3},
}

H_BASE = 3
H_EXT = 6
N_SIM = 2000
PERCENTILES = (5, 50, 95)


def errores_backtest(
    series: dict[str, pd.Series],
    cortes,
    h: int,
    ganadores: dict[str, str] = GANADORES,
) -> dict[str, np.ndarray]:
    """Errores relativos del modelo ganador por serie, con forma (ventanas, h)

    Se usan errores relativos (pred/real - 1) para que la escala del intervalo
    acompañe al nivel de precio proyectado
    """
    errores = {}
    for nombre, s in series.items():
        fn = MODELOS[ganadores[nombre]]
        filas = []
        for corte in cortes:
            tr = s[s.index <= corte]
            te = s[s.index > corte][:h]
            if len(te) < h:
                continue
            pred = fn(tr, h)
            filas.append(pred / te.to_numpy() - 1)
        errores[nombre] = np.array(filas)
    return errores


def simular(
    series: dict[str, pd.Series],
    errores: dict[str, np.ndarray],
    h: int,
    n_sim: int = N_SIM,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Trayectorias simuladas por serie, con forma (n_sim, h)

    En cada simulación se muestrea la MISMA ventana de error para las tres series,
    de modo que la correlación entre materias primas se preserva al combinarlas
    """
    rng = np.random.default_rng(seed)
    n_ventanas = min(e.shape[0] for e in errores.values())
    idx = rng.integers(0, n_ventanas, size=n_sim)
    ruido = rng.normal(0, 1, size=(n_sim, h)) * 0.3  # jitter sobre el error empírico

    sims = {}
    for nombre, s in series.items():
        fn = MODELOS[GANADORES[nombre]]
        punto = fn(s, h)
        e = errores[nombre][idx] * (1 + ruido)
        sims[nombre] = punto * (1 - e)
    return sims


def combinar_equipos(sims: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Aplica las fórmulas a cada trayectoria simulada"""
    return {
        equipo: sum(peso * sims[mp] for mp, peso in pesos.items() if peso)
        for equipo, pesos in FORMULAS.items()
    }


def resumir(sims: dict[str, np.ndarray], fechas: pd.DatetimeIndex) -> pd.DataFrame:
    """Percentiles por paso del horizonte, en formato largo"""
    filas = []
    for nombre, arr in sims.items():
        for i, fecha in enumerate(fechas):
            p5, p50, p95 = np.percentile(arr[:, i], PERCENTILES)
            filas.append({
                "serie": nombre,
                "fecha": fecha.date(),
                "h": i + 1,
                "p5": round(p5, 2),
                "p50": round(p50, 2),
                "p95": round(p95, 2),
                "ancho_rel": round((p95 - p5) / p50, 4),
            })
    return pd.DataFrame(filas)


def fechas_futuras(series: dict[str, pd.Series], h: int) -> pd.DatetimeIndex:
    origen = min(s.index.max() for s in series.values())
    return pd.date_range(origen + pd.DateOffset(months=1), periods=h, freq="ME")


def proyectar(series: dict[str, pd.Series], cortes, h: int) -> pd.DataFrame:
    """Pipeline completo: errores, simulación, combinación y percentiles"""
    errores = errores_backtest(series, cortes, h)
    sims_mp = simular(series, errores, h)
    sims_eq = combinar_equipos(sims_mp)
    fechas = fechas_futuras(series, h)
    return pd.concat([resumir(sims_mp, fechas), resumir(sims_eq, fechas)], ignore_index=True)


def main() -> None:
    from src.ingest.build import load_raw

    dfs = load_raw()
    series = {}
    for nombre, df in zip(["X", "Y", "Z"], dfs):
        series[nombre] = df.set_index("Date")[f"Price_{nombre}"].sort_index().resample("ME").mean()
    inicio = max(s.index.min() for s in series.values())
    fin = min(s.index.max() for s in series.values())
    series = {k: s[(s.index >= inicio) & (s.index <= fin)] for k, s in series.items()}

    CORTES = pd.to_datetime(SELECCION["cortes"])
    FORECASTS.mkdir(parents=True, exist_ok=True)
    for h, etiqueta in [(H_BASE, "base"), (H_EXT, "extendido")]:
        out = proyectar(series, CORTES , h)
        out.to_csv(FORECASTS / f"pronostico_{etiqueta}_{h}m.csv", index=False)
        print(f"\nHorizonte {h} meses ({etiqueta}):")
        print(out[out.serie.str.startswith("Equipo")].to_string(index=False))


if __name__ == "__main__":
    main()