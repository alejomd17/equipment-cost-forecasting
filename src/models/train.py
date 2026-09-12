"""Selección de modelo por serie y verificación de las fórmulas de los equipos

Corre el backtest multi-ventana, elige el modelo ganador de cada materia prima
según MASE en el horizonte base y guarda el resultado en models/seleccion.json,
que es lo que consume el pronóstico. También verifica que los coeficientes
estimados por regresión sigan siendo compatibles con las fórmulas asumidas
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.forecast.commodities import backtest, ranking
from src.ingest.build import load_raw

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

CORTES = ["2019-08-31", "2020-08-31", "2021-08-31", "2022-08-31"]
HORIZONTES = (3, 6, 12)
H_SELECCION = 3

# Fórmulas asumidas a partir de 02_modeling, con tolerancia para la verificación
FORMULAS = {
    "Equipo1": {"Price_X": 0.2, "Price_Y": 0.8, "Price_Z": 0.0},
    "Equipo2": {"Price_X": 1 / 3, "Price_Y": 1 / 3, "Price_Z": 1 / 3},
}
TOLERANCIA = 0.02
MEJORA_MINIMA = 0.05

def cargar_series() -> dict[str, pd.Series]:
    """Series mensuales promedio de X, Y, Z sobre el período común"""
    dfs = load_raw()
    series = {}
    for nombre, df in zip(["X", "Y", "Z"], dfs):
        series[nombre] = df.set_index("Date")[f"Price_{nombre}"].sort_index().resample("ME").mean()
    inicio = max(s.index.min() for s in series.values())
    fin = min(s.index.max() for s in series.values())
    return {k: s[(s.index >= inicio) & (s.index <= fin)] for k, s in series.items()}


def elegir_ganadores(rk: pd.DataFrame, h: int = H_SELECCION) -> dict[str, str]:
    """Gana el modelo que lidere al menos dos de las tres métricas y supere a naive por MEJORA_MINIMA"""
    sub = rk[rk["h"] == h]
    ganadores = {}
    for serie, g in sub.groupby("serie"):
        votos = pd.Series(
            [g.loc[g[met].idxmin(), "modelo"] for met in ["MAPE", "RMSE", "MASE"]]
        ).value_counts()
        lider, n = votos.index[0], votos.iloc[0]
        base = float(g[g["modelo"] == "naive"]["MASE"].iloc[0])
        mase_lider = float(g[g["modelo"] == lider]["MASE"].iloc[0])
        mejora = (base - mase_lider) / base
        ganadores[str(serie)] = str(lider) if n >= 2 and mejora >= MEJORA_MINIMA else "naive"
    return ganadores

def verificar_formulas() -> dict[str, dict]:
    """Reestima los coeficientes y compara contra las fórmulas asumidas"""
    df = pd.read_csv(ROOT / "data" / "processed" / "historico.csv", parse_dates=["Date"])
    m = df.set_index("Date").resample("ME").mean()
    cols = ["Price_X", "Price_Y", "Price_Z"]

    resultado = {}
    for equipo, pesos in FORMULAS.items():
        fit = sm.OLS(m[f"Price_{equipo}"], sm.add_constant(m[cols])).fit()
        est = {c: round(float(fit.params[c]), 4) for c in cols}
        desvios = {c: abs(est[c] - pesos[c]) for c in cols}
        ok = all(d <= TOLERANCIA for d in desvios.values())
        pred = sum(pesos[c] * m[c] for c in cols)
        mape = float(np.mean(np.abs((m[f"Price_{equipo}"] - pred) / m[f"Price_{equipo}"])))

        resultado[equipo] = {
            "asumido": {c: round(pesos[c], 4) for c in cols},
            "estimado": est,
            "desvio_max": round(max(desvios.values()), 4),
            "mape_formula": round(mape, 5),
            "compatible": ok,
        }
        if not ok:
            print(f"AVISO: los coeficientes de {equipo} se alejaron de la fórmula asumida")
    return resultado


def main() -> None:
    series = cargar_series()
    cortes = pd.to_datetime(CORTES)

    bt = backtest(series, cortes, HORIZONTES)
    rk = ranking(bt)
    ganadores = elegir_ganadores(rk)
    formulas = verificar_formulas()

    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    rk.round(4).to_csv(REPORTS / "backtest_ranking.csv", index=False)

    metricas = (
        rk[rk["h"] == H_SELECCION]
        .set_index(["serie", "modelo"])[["MAPE", "RMSE", "MASE"]]
        .round(4)
        .to_dict("index")
    )

    seleccion = {
        "generado": pd.Timestamp.today().date().isoformat(),
        "cortes": CORTES,
        "horizontes": list(HORIZONTES),
        "h_seleccion": H_SELECCION,
        "criterio": f"lidera 2 de 3 métricas y mejora al menos {MEJORA_MINIMA:.0%} el MASE de naive",
        "mejora_minima": MEJORA_MINIMA,
        "ganadores": ganadores,
        "metricas_h_seleccion": {f"{s}|{mo}": v for (s, mo), v in metricas.items()},
        "formulas_equipos": formulas,
    }
    (MODELS / "seleccion.json").write_text(json.dumps(seleccion, indent=2, ensure_ascii=False))

    print("\nGanadores por serie (h =", H_SELECCION, "):")
    for serie, modelo in ganadores.items():
        fila = rk[(rk.serie == serie) & (rk.h == H_SELECCION) & (rk.modelo == modelo)].iloc[0]
        print(f"  {serie}: {modelo}  MAPE {fila.MAPE:.4f}  MASE {fila.MASE:.4f}")

    print("\nVerificación de fórmulas:")
    for equipo, v in formulas.items():
        estado = "compatible" if v["compatible"] else "REVISAR"
        print(f"  {equipo}: {estado}  desvío máx {v['desvio_max']}  MAPE {v['mape_formula']:.5f}")
        
if __name__ == "__main__":
    main()