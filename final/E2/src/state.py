# Estado compartido del grafo multi-agente - E2 con Critic
from typing import TypedDict, List, Optional, Annotated
from operator import add


class AgentState(TypedDict):
    """Estado que fluye entre nodos del grafo."""
    question: str  # Pregunta original
    subtasks: List[str]  # Subtareas generadas por Manager
    context: List[str]  # Contexto recuperado por RAG
    partial_answers: Annotated[List[str], add]  # Respuestas de Workers
    final_answer: Optional[str]  # Respuesta consolidada
    critic_review: Optional[dict]  # Revisión del Critic
    logs: Annotated[List[dict], add]  # Trazas de ejecución
