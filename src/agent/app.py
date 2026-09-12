"""Interfaz de chat para el agente

Streamlit reejecuta el script completo en cada interacción, así que el grafo se
construye una sola vez con cache_resource y el historial vive en session_state
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.graph import construir_agente, preguntar
import streamlit as st

HILO = "streamlit"

EJEMPLOS = [
    "¿Cuánto va a costar el Equipo 1 en los próximos tres meses?",
    "¿Por qué el Equipo 2 tiene menos incertidumbre?",
    "¿Qué pasa si el precio de Y sube 15%?",
    "Necesito comprar 2 unidades del Equipo 1 en septiembre y 1 del Equipo 2 en noviembre, ¿cuánto presupuesto?",
    "¿Cómo viene el mercado de commodities de construcción este año?",
]

st.set_page_config(page_title="Asistente de costos de equipos", page_icon="📊", layout="centered")


@st.cache_resource
def cargar_agente():
    return construir_agente()


with st.sidebar:
    st.subheader("Sobre este asistente")
    st.write(
        "Responde sobre el pronóstico de costos de dos equipos de construcción. "
        "Toda cifra que menciona proviene de los artefactos del análisis, no del modelo."
    )
    st.caption(
        "Los equipos son combinaciones lineales de tres materias primas anonimizadas. "
        "El horizonte recomendado de planeación es de tres meses."
    )
    st.divider()
    st.subheader("Preguntas de ejemplo")
    for i, ej in enumerate(EJEMPLOS):
        if st.button(ej, key=f"ej{i}", use_container_width=True):
            st.session_state.pendiente = ej
    st.divider()
    if st.button("Reiniciar conversación", use_container_width=True):
        st.session_state.historial = []
        st.rerun()

st.title("Asistente de costos de equipos")

if "historial" not in st.session_state:
    st.session_state.historial = []

for rol, texto in st.session_state.historial:
    with st.chat_message(rol):
        st.markdown(texto)

entrada = st.chat_input("Pregunta sobre el pronóstico, los modelos o el mercado")
if "pendiente" in st.session_state:
    entrada = st.session_state.pop("pendiente")

if entrada:
    st.session_state.historial.append(("user", entrada))
    with st.chat_message("user"):
        st.markdown(entrada)

    with st.chat_message("assistant"):
        with st.spinner("Consultando el análisis..."):
            try:
                respuesta = preguntar(cargar_agente(), entrada, hilo=HILO)
            except Exception as e:
                respuesta = f"Ocurrió un error al procesar la consulta: {e}"
        st.markdown(respuesta)

    st.session_state.historial.append(("assistant", respuesta))