# Pronóstico de costos de equipos de construcción

Proyección del costo de adquisición de dos equipos a partir del precio de tres
materias primas, con intervalos de confianza y un agente conversacional que
permite consultar los resultados en lenguaje natural.

El análisis identifica que los precios de los equipos son combinaciones lineales
exactas de las materias primas, con un error inferior al 0.5%, por lo que el
problema de proyectar su costo se reduce a proyectar los insumos y aplicar esas
composiciones.

## Resultado principal

Cada equipo depende de los insumos con pesos distintos, lo que explica que uno
sea sensiblemente más predecible que el otro.

![Equipos frente a su insumo dominante](reports/figures/equipos_vs_insumo.png)

El pronóstico se entrega con intervalo de confianza al 90%. La línea punteada
marca el fin del horizonte recomendado de planeación.

![Pronóstico con intervalo](reports/figures/pronostico_equipos.png)

## Requisitos

- [uv](https://docs.astral.sh/uv/) para gestionar el entorno
- Python 3.12 (uv lo instala si no está)
- `make` es opcional: cada atajo tiene su comando equivalente
- En Linux, LightGBM necesita `libgomp1` (`sudo apt install libgomp1`). Si no
  está disponible, el pipeline omite ese modelo automáticamente y continúa
- Para el agente: una clave de [Google AI Studio](https://aistudio.google.com) y
  otra de [Tavily](https://tavily.com), ambas con plan gratuito

## Puesta en marcha

```bash
git clone https://github.com/alejomd17/equipment-cost-forecasting.git
cd equipment-cost-forecasting
uv sync
cp .env.example .env    # y completar las dos claves
```

Ejecutar el pipeline completo:

```bash
make all
```

Equivalente sin `make`:

```bash
uv run python -m src.ingest.build      # normaliza los crudos y valida
uv run python -m src.models.train      # backtest y selección de modelo
uv run python -m src.forecast.run      # proyección con intervalos
```

Levantar el agente:

```bash
make agent
# o bien
uv run streamlit run src/agent/app.py
```

Con Docker, sin instalar Python ni dependencias:

```bash
make docker-build
make docker-run
```

El contenedor lee las claves del `.env`, así que hay que crearlo igual antes de correr.

## Qué hace cada etapa

**`src/ingest/build.py`** lee los cuatro archivos crudos corrigiendo el formato
propio de cada uno (separadores, coma decimal, orden invertido, BOM), los une por
fecha y verifica que la reconstrucción coincida con el histórico entregado. La
validación imprime el porcentaje de coincidencia por serie.

**`src/models/train.py`** evalúa siete modelos candidatos sobre nueve ventanas de
backtest y tres horizontes, con MAPE, RMSE y MASE. Selecciona el modelo de cada
materia prima exigiendo que lidere al menos dos métricas y que supere al
benchmark ingenuo por un margen mínimo. También reestima los coeficientes de las
composiciones y avisa si se alejaron de los valores asumidos. El resultado queda
en `models/seleccion.json`.

**`src/forecast/run.py`** proyecta cada materia prima con su modelo ganador y
construye los intervalos por bootstrap de los errores observados en el backtest,
sin supuestos distribucionales. Las trayectorias simuladas se combinan con las
composiciones para obtener el costo de cada equipo.

**`src/agent/`** expone los artefactos como herramientas de un agente ReAct sobre
LangGraph. El agente consulta el pronóstico, explica cómo se construyó, simula
escenarios de shock en los insumos, calcula presupuestos de compra, evalúa
cotizaciones de proveedores contra el rango proyectado y busca contexto de
mercado en la web.

![El agente evaluando una cotización de proveedor](reports/figures/agente_cotizacion.png)

## Estructura

```
data/raw/              archivos originales, sin modificar
data/processed/        histórico unificado
data/forecasts/        pronósticos con intervalos
models/                selección de modelo y métricas
notebooks/             exploración, modelado y proyección
reports/               informe, tablas y figuras
docs/                  diagramas de arquitectura
src/ingest/            lectura y validación de los crudos
src/features/          transformaciones compartidas
src/models/            backtest y selección
src/forecast/          modelos candidatos y proyección
src/agent/             herramientas, grafo e interfaz
```

Los notebooks importan de `src/` en lugar de duplicar lógica, de modo que lo que
corre en `make all` es el mismo código que produjo las figuras.

## Arquitectura

- [Arquitectura implementada](docs/arquitectura_local.svg)
- [Arquitectura propuesta en Azure](docs/arquitectura_azure.svg)

## Informe

El análisis completo está en [`reports/informe.md`](reports/informe.md), también en
[PDF](reports/Informe_pronostico_costos_equipos.pdf). La
[presentación](reports/Presentacion_pronostico_costos_equipos.pptx) resume los
hallazgos en diapositivas.