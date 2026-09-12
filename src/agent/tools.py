"""Herramientas que el agente puede invocar

Cada función expuesta con @tool se convierte en una herramienta que el modelo
puede elegir. El modelo no ve el código: ve el nombre, la firma con sus tipos
y el docstring. Por eso la descripción es parte del diseño, no documentación
opcional, porque es lo único con lo que el modelo decide cuándo usarla y con
qué argumentos

Las herramientas se agrupan en tres familias: lectura de los artefactos que
produjo el pipeline, cálculo sobre las fórmulas validadas, y contexto externo
de mercado. Ninguna inventa datos: si falta un artefacto lo dicen, para que el
agente pueda explicarle al usuario qué hace falta ejecutar
"""

import json
import os
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

ROOT = Path(__file__).resolve().parents[2]
FORECASTS = ROOT / "data" / "forecasts"
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"

EQUIPOS = {"1": "Equipo1", "2": "Equipo2"}

# Composición validada en 02_modeling, replicada aquí para el cálculo de escenarios
FORMULAS = {
    "Equipo1": {"X": 0.2, "Y": 0.8, "Z": 0.0},
    "Equipo2": {"X": 1 / 3, "Y": 1 / 3, "Z": 1 / 3},
}


def _leer_pronostico(horizonte: int) -> tuple[pd.DataFrame | None, int]:
    """Carga el CSV de pronóstico del escenario que corresponda al horizonte"""
    etiqueta = "base" if horizonte <= 3 else "extendido"
    h = 3 if horizonte <= 3 else 6
    path = FORECASTS / f"pronostico_{etiqueta}_{h}m.csv"
    return (pd.read_csv(path) if path.exists() else None), h


@tool
def consultar_pronostico(equipo: str = "ambos", horizonte: int = 3) -> str:
    """Devuelve el pronóstico de costo con su intervalo de confianza al 90%

    Úsala cuando pregunten cuánto va a costar un equipo, qué proyección hay
    para un mes concreto, o cuál es el rango esperado de precios

    Args:
        equipo: "1", "2" o "ambos"
        horizonte: 3 para el escenario base recomendado, 6 para el extendido
    """
    df, h = _leer_pronostico(horizonte)
    if df is None:
        return "No hay pronóstico generado. Ejecutar 'make forecast' primero"

    df = df[df["serie"].str.startswith("Equipo")]
    if equipo in EQUIPOS:
        df = df[df["serie"] == EQUIPOS[equipo]]

    lineas = [f"Pronóstico a {h} meses (mediana e intervalo de confianza al 90%):"]
    for _, r in df.iterrows():
        lineas.append(
            f"{r['serie']} | {r['fecha']} | mediana {r['p50']:.0f} | "
            f"rango [{r['p5']:.0f}, {r['p95']:.0f}] | amplitud relativa {r['ancho_rel']:.1%}"
        )
    lineas.append(
        "El horizonte recomendado para planeación es de 3 meses. Más allá, la amplitud "
        "del intervalo supera el 40% y deja de ser útil para presupuestar"
    )
    return "\n".join(lineas)


@tool
def resumen_analisis() -> str:
    """Devuelve cómo se construyó el pronóstico: composición de los equipos,
    modelo elegido por materia prima y métricas de validación

    Úsala cuando pregunten de qué depende el costo de un equipo, qué modelo se
    usó, qué tan confiable es, o por qué se eligió un método sobre otro
    """
    path = MODELS / "seleccion.json"
    if not path.exists():
        return "No hay selección de modelo registrada. Ejecutar 'make train' primero"

    sel = json.loads(path.read_text())

    lineas = ["Composición de los equipos (relación determinística validada por regresión):"]
    for equipo, info in sel["formulas_equipos"].items():
        pesos = ", ".join(
            f"{k.replace('Price_', '')}={v}" for k, v in info["asumido"].items() if v
        )
        lineas.append(f"  {equipo} = {pesos} | error de la fórmula {info['mape_formula']:.2%}")

    lineas.append("\nModelo seleccionado por materia prima:")
    for serie, modelo in sel["ganadores"].items():
        met = sel["metricas_h_seleccion"].get(f"{serie}|{modelo}", {})
        mape, mase = met.get("MAPE"), met.get("MASE")
        detalle = f" | MAPE {mape:.2%} | MASE {mase:.2f}" if mape is not None else ""
        lineas.append(f"  {serie}: {modelo}{detalle}")

    lineas.append(f"\nCriterio de selección: {sel['criterio']}")
    lineas.append(
        f"Validado sobre {len(sel['cortes'])} ventanas de backtest entre "
        f"{sel['cortes'][0][:4]} y {sel['cortes'][-1][:4]}"
    )
    return "\n".join(lineas)


@tool
def historico_precios(serie: str, meses: int = 12) -> str:
    """Devuelve los últimos meses observados de una serie

    Úsala cuando pregunten cómo venía comportándose un precio, si subió o bajó
    recientemente, o para comparar el pronóstico contra lo que ya ocurrió

    Args:
        serie: "X", "Y", "Z", "Equipo1" o "Equipo2"
        meses: cuántos meses hacia atrás devolver
    """
    path = PROCESSED / "historico.csv"
    if not path.exists():
        return "No hay histórico procesado. Ejecutar 'make data' primero"

    col = f"Price_{serie}"
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    if col not in df.columns:
        disponibles = ", ".join(c.replace("Price_", "") for c in df.columns)
        return f"Serie desconocida. Opciones: {disponibles}"

    s = df[col].resample("ME").mean().tail(meses)
    variacion = (s.iloc[-1] / s.iloc[0] - 1) if len(s) > 1 else 0
    lineas = [f"Últimos {len(s)} meses de {serie} (promedio mensual):"]
    lineas += [f"  {f.date()}: {v:.1f}" for f, v in s.items()]
    lineas.append(f"Variación en el período: {variacion:+.1%}")
    return "\n".join(lineas)


@tool
def analisis_sensibilidad(materia_prima: str, shock_pct: float, horizonte: int = 3) -> str:
    """Calcula cómo cambia el costo de cada equipo si una materia prima varía

    Como la composición de los equipos es una combinación lineal exacta de las
    materias primas, el impacto se calcula de forma determinística, sin
    reestimar el modelo

    Úsala para preguntas del tipo "qué pasa si X sube 15%" o "cuánto me afecta
    una caída de Y"

    Args:
        materia_prima: "X", "Y" o "Z"
        shock_pct: variación porcentual, por ejemplo 15 para +15% o -10 para una caída
        horizonte: 3 o 6 meses
    """
    mp = materia_prima.upper().replace("PRICE_", "")
    if mp not in ("X", "Y", "Z"):
        return "Materia prima desconocida. Opciones: X, Y, Z"

    df, h = _leer_pronostico(horizonte)
    if df is None:
        return "No hay pronóstico generado. Ejecutar 'make forecast' primero"

    base_mp = df[df["serie"] == mp].set_index("h")["p50"]
    factor = shock_pct / 100

    lineas = [f"Impacto de una variación de {shock_pct:+.0f}% en {mp}, a {h} meses:"]
    for equipo, pesos in FORMULAS.items():
        peso = pesos[mp]
        if peso == 0:
            lineas.append(f"\n{equipo}: no depende de {mp}, el impacto es nulo")
            continue
        eq = df[df["serie"] == equipo].set_index("h")
        lineas.append(f"\n{equipo} (peso de {mp}: {peso:.2f}):")
        for paso in eq.index:
            delta = peso * base_mp.loc[paso] * factor
            nuevo = eq.loc[paso, "p50"] + delta
            rel = delta / eq.loc[paso, "p50"]
            lineas.append(
                f"  {eq.loc[paso, 'fecha']}: {eq.loc[paso, 'p50']:.0f} -> {nuevo:.0f} "
                f"({rel:+.1%})"
            )
    return "\n".join(lineas)


@tool
def comparar_equipos(horizonte: int = 3) -> str:
    """Compara los dos equipos en nivel de costo y en incertidumbre

    Úsala cuando pregunten cuál equipo es más caro, cuál tiene más riesgo de
    precio, o por qué uno es más predecible que el otro

    Args:
        horizonte: 3 o 6 meses
    """
    df, h = _leer_pronostico(horizonte)
    if df is None:
        return "No hay pronóstico generado. Ejecutar 'make forecast' primero"

    e1 = df[df["serie"] == "Equipo1"].set_index("h")
    e2 = df[df["serie"] == "Equipo2"].set_index("h")

    lineas = [f"Comparación de los dos equipos a {h} meses:"]
    for paso in e1.index:
        a, b = e1.loc[paso], e2.loc[paso]
        dif = b["p50"] - a["p50"]
        lineas.append(
            f"  {a['fecha']}: Equipo1 {a['p50']:.0f} (amplitud {a['ancho_rel']:.1%}) | "
            f"Equipo2 {b['p50']:.0f} (amplitud {b['ancho_rel']:.1%}) | "
            f"diferencia {dif:+.0f}"
        )
    lineas.append(
        "\nEl Equipo 2 presenta menor incertidumbre relativa porque promedia las tres "
        "materias primas en partes iguales, y la diversificación reduce la varianza. "
        "El Equipo 1 concentra el 80% de su composición en una sola serie, por lo que "
        "hereda casi toda la volatilidad de esa materia prima"
    )
    return "\n".join(lineas)


@tool
def presupuesto_proyecto(compras: str, percentil: str = "p95") -> str:
    """Calcula el costo total de un calendario de compras con su rango

    Úsala cuando pregunten cuánto presupuestar para adquirir varias unidades en
    distintos meses. Para presupuesto conservador conviene el percentil 95, que
    cubre el escenario alto del intervalo

    Args:
        compras: lista en formato "equipo:cantidad:mes", separada por comas.
            El mes es 1, 2 o 3 dentro del horizonte. Ejemplo: "1:2:1, 2:1:3"
        percentil: "p5", "p50" o "p95" para el escenario bajo, central o alto
    """
    if percentil not in ("p5", "p50", "p95"):
        return "Percentil no válido. Opciones: p5, p50, p95"

    df, h = _leer_pronostico(6)
    if df is None:
        return "No hay pronóstico generado. Ejecutar 'make forecast' primero"

    total = 0.0
    lineas = [f"Presupuesto con escenario {percentil}:"]
    for item in compras.split(","):
        partes = [p.strip() for p in item.split(":")]
        if len(partes) != 3:
            return 'Formato no válido. Usar "equipo:cantidad:mes", por ejemplo "1:2:1, 2:1:3"'
        eq_id, cantidad, mes = partes
        if eq_id not in EQUIPOS:
            return f"Equipo desconocido: {eq_id}. Opciones: 1, 2"

        fila = df[(df["serie"] == EQUIPOS[eq_id]) & (df["h"] == int(mes))]
        if fila.empty:
            return f"El mes {mes} está fuera del horizonte disponible ({h} meses)"

        unitario = float(fila.iloc[0][percentil])
        subtotal = unitario * int(cantidad)
        total += subtotal
        lineas.append(
            f"  {EQUIPOS[eq_id]} x{cantidad} en {fila.iloc[0]['fecha']}: "
            f"{unitario:.0f} c/u = {subtotal:,.0f}"
        )

    lineas.append(f"\nTotal: {total:,.0f}")
    if percentil == "p95":
        lineas.append(
            "Este total corresponde al escenario alto. Con el percentil 50 se obtiene el "
            "costo esperado y con el 5 el escenario favorable"
        )
    return "\n".join(lineas)


@tool
def estado_pipeline() -> str:
    """Informa qué artefactos del análisis existen y hasta qué fecha llegan los datos

    Úsala cuando pregunten si el análisis está actualizado, de cuándo son los
    datos, o cuando una consulta falle por falta de algún archivo
    """
    lineas = ["Estado del pipeline:"]

    hist = PROCESSED / "historico.csv"
    if hist.exists():
        df = pd.read_csv(hist, parse_dates=["Date"])
        lineas.append(
            f"  Histórico procesado: {len(df)} registros diarios, "
            f"de {df['Date'].min().date()} a {df['Date'].max().date()}"
        )
    else:
        lineas.append("  Histórico procesado: no generado (ejecutar 'make data')")

    sel = MODELS / "seleccion.json"
    if sel.exists():
        info = json.loads(sel.read_text())
        lineas.append(
            f"  Selección de modelo: generada el {info['generado']}, "
            f"ganadores {info['ganadores']}"
        )
    else:
        lineas.append("  Selección de modelo: no generada (ejecutar 'make train')")

    encontrados = sorted(p.name for p in FORECASTS.glob("pronostico_*.csv"))
    if encontrados:
        lineas.append(f"  Pronósticos disponibles: {', '.join(encontrados)}")
    else:
        lineas.append("  Pronósticos: no generados (ejecutar 'make forecast')")

    return "\n".join(lineas)


@tool
def buscar_contexto_mercado(consulta: str) -> str:
    """Busca noticias y análisis recientes del mercado en la web

    Las materias primas del análisis están anonimizadas como X, Y y Z, por lo que
    esta herramienta no puede traer noticias sobre ellas en particular. Sirve para
    contexto general: sector construcción, mercados de commodities, condiciones
    macroeconómicas, o cualquier tema que el usuario mencione explícitamente

    Úsala cuando la pregunta requiera información externa al análisis. No la uses
    para consultar cifras del pronóstico, que viven en las otras herramientas, ni
    para intentar averiguar qué son X, Y o Z

    Args:
        consulta: qué buscar, en lenguaje natural
    """
    if not os.getenv("TAVILY_API_KEY"):
        return "Búsqueda web no disponible: falta configurar TAVILY_API_KEY"

    try:
        buscador = TavilySearch(max_results=4, topic="news")
        resultados = buscador.invoke({"query": consulta})
    except Exception as e:
        return f"La búsqueda web falló: {e}"

    items = resultados.get("results", []) if isinstance(resultados, dict) else []
    if not items:
        return "La búsqueda no devolvió resultados relevantes"

    lineas = [
        "Contexto externo de mercado (fuente web, no vinculado a las series X, Y, Z "
        "del análisis interno):"
    ]
    for r in items:
        lineas.append(f"\n{r.get('title', 'sin título')} ({r.get('url', '')})")
        lineas.append(r.get("content", "")[:400])
    return "\n".join(lineas)


@tool
def evaluar_cotizacion(equipo: str, precio_ofertado: float, mes: int = 1) -> str:
    """Compara el precio que ofrece un proveedor contra el rango proyectado

    Permite juzgar una cotización de forma objetiva: si cae por debajo del
    percentil 5 es una oportunidad frente a lo que anticipa el mercado, si cae
    dentro del rango es un precio normal, y si supera el percentil 95 está por
    encima de lo esperado. Sirve también para comparar varias cotizaciones entre
    sí usando la misma referencia

    Úsala cuando mencionen un precio de proveedor, una cotización recibida, o
    pregunten si un precio es razonable

    Args:
        equipo: "1" o "2"
        precio_ofertado: precio por unidad que ofrece el proveedor
        mes: mes del horizonte al que aplica la cotización (1, 2 o 3)
    """
    if equipo not in EQUIPOS:
        return "Equipo desconocido. Opciones: 1, 2"

    df, h = _leer_pronostico(6)
    if df is None:
        return "No hay pronóstico generado. Ejecutar 'make forecast' primero"

    fila = df[(df["serie"] == EQUIPOS[equipo]) & (df["h"] == int(mes))]
    if fila.empty:
        return f"El mes {mes} está fuera del horizonte disponible ({h} meses)"

    r = fila.iloc[0]
    p5, p50, p95 = float(r["p5"]), float(r["p50"]), float(r["p95"])
    desviacion = (precio_ofertado - p50) / p50

    if precio_ofertado < p5:
        veredicto = "Por debajo del rango proyectado: oportunidad frente a lo que anticipa el mercado"
    elif precio_ofertado > p95:
        veredicto = "Por encima del rango proyectado: precio alto frente a lo que anticipa el mercado"
    else:
        veredicto = "Dentro del rango proyectado: precio consistente con el mercado esperado"

    return "\n".join([
        f"Evaluación de la cotización para {EQUIPOS[equipo]} en {r['fecha']}:",
        f"  Precio ofertado: {precio_ofertado:,.0f}",
        f"  Rango proyectado al 90%: [{p5:,.0f}, {p95:,.0f}] con mediana {p50:,.0f}",
        f"  Desviación frente a la mediana: {desviacion:+.1%}",
        f"\n{veredicto}",
        "\nEl rango proviene del pronóstico interno y no incluye condiciones comerciales "
        "como plazos de entrega, garantías o volumen, que deben pesar en la decisión final",
    ])


TOOLS = [
    consultar_pronostico,
    resumen_analisis,
    historico_precios,
    analisis_sensibilidad,
    comparar_equipos,
    presupuesto_proyecto,
    estado_pipeline,
    buscar_contexto_mercado,
    evaluar_cotizacion,
]