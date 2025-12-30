"""
Módulo de gestión de estado.

Proporciona seguimiento del estado de procesamiento y gestión de estado de fallos.
"""

from state.processing_state import ProcessingState, get_processing_state

__all__ = [
    "ProcessingState",
    "get_processing_state",
]
