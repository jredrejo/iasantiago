"""
Módulo de gestión de estado.

Proporciona seguimiento del estado de procesamiento y gestión de estado de fallos.
"""

from state.inflight import InflightTracker, get_inflight_tracker
from state.processing_state import ProcessingState, get_processing_state

__all__ = [
    "InflightTracker",
    "ProcessingState",
    "get_inflight_tracker",
    "get_processing_state",
]
