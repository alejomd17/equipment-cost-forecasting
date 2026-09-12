"""Modelos candidatos para pronosticar las series de materias primas.

Cada función recibe la serie de entrenamiento y el horizonte, y devuelve un
array con las predicciones. El diccionario MODELOS los expone por nombre para
que el backtest y el pronóstico final los recorran sin repetir código.
"""

import logging

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from lightgbm import LGBMRegressor
    LGBM_DISPONIBLE = True
except (ImportError, OSError):
    LGBM_DISPONIBLE = False
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

LAGS = (1, 2, 3, 6, 12)


def pred_naive(tr: pd.Series, h: int) -> np.ndarray:
    """Último valor observado repetido. Benchmark para series con raíz unitaria."""
    return np.repeat(tr.iloc[-1], h)


def pred_drift(tr: pd.Series, h: int) -> np.ndarray:
    """Random walk con drift: último valor más la pendiente promedio histórica."""
    pendiente = (tr.iloc[-1] - tr.iloc[0]) / (len(tr) - 1)
    return tr.iloc[-1] + pendiente * np.arange(1, h + 1)


def pred_arima(tr: pd.Series, h: int, order=(1, 1, 1)) -> np.ndarray:
    return ARIMA(tr, order=order).fit().forecast(h).values


def pred_sarima(tr: pd.Series, h: int, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12)) -> np.ndarray:
    fit = SARIMAX(tr, order=order, seasonal_order=seasonal_order).fit(disp=False)
    return fit.forecast(h).values


def pred_ets(tr: pd.Series, h: int) -> np.ndarray:
    return ExponentialSmoothing(tr, trend="add").fit().forecast(h).values


def pred_prophet(tr: pd.Series, h: int) -> np.ndarray:
    d = tr.reset_index()
    d.columns = ["ds", "y"]
    mdl = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    mdl.fit(d)
    fut = mdl.make_future_dataframe(periods=h, freq="ME")
    return mdl.predict(fut)["yhat"].tail(h).to_numpy()


def _make_features(s: pd.Series, lags=LAGS) -> pd.DataFrame:
    """Rezagos, medias móviles y mes del año como variables explicativas."""
    d = pd.DataFrame({"y": s})
    for lag in lags:
        d[f"lag{lag}"] = d["y"].shift(lag)
    d["ma3"] = d["y"].shift(1).rolling(3).mean()
    d["ma12"] = d["y"].shift(1).rolling(12).mean()
    d["mes"] = d.index.month
    return d.dropna()


def pred_lgbm(tr: pd.Series, h: int, lags=LAGS) -> np.ndarray:
    """Predicción recursiva: cada mes predicho alimenta los rezagos del siguiente."""
    d = _make_features(tr, lags)
    mdl = LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1)
    mdl.fit(d.drop(columns="y"), d["y"])
    cols = d.drop(columns="y").columns

    hist = tr.copy()
    preds = []
    for _ in range(h):
        fecha = hist.index[-1] + pd.DateOffset(months=1)
        fila = {f"lag{lag}": hist.iloc[-lag] for lag in lags}
        fila["ma3"] = hist.iloc[-3:].mean()
        fila["ma12"] = hist.iloc[-12:].mean()
        fila["mes"] = fecha.month
        p = mdl.predict(pd.DataFrame([fila])[cols])[0]
        preds.append(p)
        hist = pd.concat([hist, pd.Series([p], index=[fecha])])
    return np.array(preds)


MODELOS = {
    "naive": pred_naive,
    "drift": pred_drift,
    "arima": pred_arima,
    "sarima": pred_sarima,
    "ets": pred_ets,
    "prophet": pred_prophet,
    "lightgbm": pred_lgbm,
}

if not LGBM_DISPONIBLE:
    MODELOS.pop("lightgbm")

def mase(train: np.ndarray, real: np.ndarray, pred: np.ndarray) -> float:
    """Error absoluto medio escalado por el del naive en entrenamiento (Hyndman & Koehler)."""
    escala = np.mean(np.abs(np.diff(train)))
    return np.mean(np.abs(real - pred)) / escala

def backtest(
    series: dict[str, pd.Series],
    cortes: list,
    horizontes: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """Evalúa cada modelo en varias ventanas y horizontes. MAPE, RMSE y MASE."""
    filas = []
    for corte in cortes:
        for nombre, s in series.items():
            tr = s[s.index <= corte]
            futuro = s[s.index > corte]
            for h in horizontes:
                te = futuro[:h]
                if len(te) < h:
                    continue
                for modelo, fn in MODELOS.items():
                    p = fn(tr, h)
                    real = te.to_numpy()
                    filas.append({
                        "corte": pd.Timestamp(corte).date(),
                        "serie": nombre,
                        "h": h,
                        "modelo": modelo,
                        "MAPE": np.mean(np.abs((real - p) / real)),
                        "RMSE": np.sqrt(np.mean((real - p) ** 2)),
                        "MASE": mase(tr.to_numpy(), real, p),
                    })
    return pd.DataFrame(filas)


def ranking(bt: pd.DataFrame) -> pd.DataFrame:
    """Promedia métricas sobre todas las ventanas y marca el ganador por serie y horizonte."""
    agg = bt.groupby(["serie", "h", "modelo"])[["MAPE", "RMSE", "MASE"]].mean().reset_index()
    agg["ganador"] = agg["MASE"] == agg.groupby(["serie", "h"])["MASE"].transform("min")
    return agg.sort_values(["serie", "h", "MASE"])