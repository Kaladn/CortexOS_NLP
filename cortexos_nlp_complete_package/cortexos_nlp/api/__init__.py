"""
CortexOS NLP - API Layer Package
Phase 3: Developer-Friendly Interface

This package provides the spaCy-compatible API layer that makes
the CortexOS Deterministic NLP Engine accessible to developers worldwide.
"""

from .cortex_nlp import CortexNLP
from .cortex_doc import Doc
from .cortex_token import Token
from .cortex_span import Span

__all__ = [
    'CortexNLP',
    'Doc', 
    'Token',
    'Span'
]

__version__ = "1.0.0"
__author__ = "CortexOS Team"
__description__ = "Deterministic NLP Engine with Mathematical Certainty"

