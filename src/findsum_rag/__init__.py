"""Avaliacao de RAG e few-shot prompting na sumarizacao de relatorios financeiros."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import DEFAULT_ARMS, ExperimentArm, ExperimentConfig
from .data import Document, Segment, Table, Task, load_documents

__all__ = [
    "DEFAULT_ARMS",
    "Document",
    "ExperimentArm",
    "ExperimentConfig",
    "Segment",
    "Table",
    "Task",
    "load_documents",
]
