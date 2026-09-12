"""Grafo del agente con LangGraph

El agente sigue el patrón ReAct: recibe la pregunta, razona sobre qué
herramienta necesita, la invoca, lee el resultado y decide si ya puede
responder o si le falta información. Ese ciclo puede repetirse varias veces
en un mismo turno, que es lo que le permite combinar el pronóstico interno
con contexto externo sin que nadie le diga el orden

El grafo tiene dos nodos y una arista condicional:

    agente --(pidió herramientas?)--> herramientas --> agente
       |
       +--(no)--> fin

El estado acumula los mensajes del turno y, con el checkpointer, también los
de turnos anteriores, que es lo que le da memoria conversacional
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.tools import TOOLS

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODELO = "gemini-3.7-flash"
MAX_ITERACIONES = 8

SYSTEM_PROMPT = """Eres un asistente analítico para el área de compras de una empresa
constructora. Tu función es explicar y hacer accesible el análisis de costos de dos equipos
cuya adquisición está planeada.

Contexto del análisis que ya está hecho:
- Los precios de los dos equipos son combinaciones lineales exactas de tres materias primas
  anonimizadas como X, Y y Z. Esa relación se validó por regresión y tiene un error inferior
  al 0.5%.
- La materia prima X corresponde al petróleo Brent, verificado por correlación de 1.00
  contra la referencia pública en niveles y en retornos. Pesa 20% en el Equipo 1 y 33%
  en el Equipo 2, así que el mercado de crudo afecta directamente ambos costos. Y y Z
  permanecen sin identificar.
- El pronóstico se construye proyectando cada materia prima con su propio modelo y aplicando
  después las fórmulas. Los intervalos salen de simular trayectorias con los errores que los
  modelos cometieron en validación histórica.
- El horizonte recomendado para planeación es de tres meses. Existe un escenario de seis meses,
  pero su intervalo es demasiado amplio para presupuestar.

Cómo trabajas:
- Toda cifra que menciones debe venir de una herramienta. Nunca inventes ni estimes números
  de memoria. Si una herramienta no devuelve el dato, dilo con claridad.
- Distingue siempre entre lo que viene del análisis interno y lo que viene de búsqueda web.
  Para X puedes buscar contexto real del mercado de crudo. Para Y y Z, que siguen
  anonimizadas, la búsqueda solo aporta contexto general del sector.
- Puedes encadenar varias herramientas en una misma respuesta cuando la pregunta lo requiera.
- Cuando cites un pronóstico, menciona el intervalo, no solo la mediana. Un valor puntual sin
  su rango induce a error en decisiones de presupuesto.
- Si te preguntan algo fuera del alcance del análisis (proveedores, contratos, otros equipos),
  dilo sin rodeos en lugar de improvisar.

Cómo respondes:
- En español, directo y sin relleno.
- Cifras con su unidad y su contexto temporal.
- Si detectas que la pregunta esconde una decisión de negocio, señala la implicación práctica.
"""


def construir_agente(con_memoria: bool = True):
    """Arma el grafo y lo compila

    Args:
        con_memoria: si es True, añade un checkpointer en memoria para que el
            agente recuerde los turnos anteriores de la misma conversación
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Falta GOOGLE_API_KEY en el archivo .env")

    # bind_tools le entrega al modelo el esquema de cada herramienta: nombre,
    # argumentos y descripción. El modelo no ejecuta nada, solo decide cuál
    # pedir y con qué argumentos
    llm = ChatGoogleGenerativeAI(model=MODELO).bind_tools(TOOLS)

    def nodo_agente(state: MessagesState) -> dict:
        """Razona sobre el estado actual y decide si responder o pedir una herramienta"""
        mensajes = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(mensajes)]}

    def hay_que_usar_herramientas(state: MessagesState) -> str:
        """Arista condicional: mira si el último mensaje pidió herramientas"""
        ultimo = state["messages"][-1]
        if getattr(ultimo, "tool_calls", None):
            return "herramientas"
        return END

    grafo = StateGraph(MessagesState)
    grafo.add_node("agente", nodo_agente)
    grafo.add_node("herramientas", ToolNode(TOOLS))

    grafo.add_edge(START, "agente")
    grafo.add_conditional_edges("agente", hay_que_usar_herramientas, ["herramientas", END])
    grafo.add_edge("herramientas", "agente")

    checkpointer = MemorySaver() if con_memoria else None
    return grafo.compile(checkpointer=checkpointer)


def preguntar(agente, pregunta: str, hilo: str = "default") -> str:
    """Envía una pregunta al agente y devuelve el texto de la respuesta"""
    config = {"configurable": {"thread_id": hilo}, "recursion_limit": MAX_ITERACIONES * 2}
    resultado = agente.invoke({"messages": [("user", pregunta)]}, config=config)
    return _texto(resultado["messages"][-1].content)

def _texto(content) -> str:
    """Gemini devuelve el contenido como lista de bloques; extrae solo el texto"""
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))

if __name__ == "__main__":
    import sys

    demos = {
        "interno": [
            "¿Cuánto va a costar el Equipo 1 en los próximos meses?",
            "¿Y por qué ese rango es tan amplio comparado con el otro equipo?",
        ],
        "externo": [
            "¿Qué está pasando en el mercado de commodities de construcción este año?",
            "¿Y eso cómo se relaciona con el pronóstico del Equipo 2?",
        ],    
        "brent": [
            "¿Qué está pasando con el petróleo y cómo afecta el costo de mis equipos?",
        ],
    }

    elegidas = sys.argv[1:] or list(demos)
    agente = construir_agente()

    for hilo in elegidas:
        if hilo not in demos:
            print(f"Sesión desconocida: {hilo}. Opciones: {', '.join(demos)}")
            continue
        print(f"\n{'=' * 60}\nSesión: {hilo}\n{'=' * 60}")
        for p in demos[hilo]:
            print(f"\n>>> {p}\n")
            print(preguntar(agente, p, hilo=hilo))