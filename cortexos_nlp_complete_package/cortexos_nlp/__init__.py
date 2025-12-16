"""
CortexOS NLP - Deterministic Natural Language Processing Engine

A revolutionary NLP engine that provides mathematical certainty instead of
probabilistic guessing, with complete spaCy compatibility for easy adoption.

Key Features:
- Mathematical certainty in all processing steps
- Complete determinism (same input = same output)
- spaCy-compatible API for easy migration
- Spatial anchor-based word representation
- Binary cell memory for relationship storage
- Harmonic resonance for similarity calculations

Example Usage:
    >>> import cortexos_nlp as spacy  # Drop-in replacement for spaCy
    >>> nlp = spacy.load("en_core_web_sm")
    >>> doc = nlp("The quick brown fox jumps over the lazy dog.")
    >>> for token in doc:
    ...     print(token.text, token.pos_, token.dep_)

Advanced Usage:
    >>> # Access CortexOS-specific features
    >>> print(token.pos_confidence)  # Mathematical confidence score
    >>> print(doc.spatial_anchors)   # 6D spatial coordinates
    >>> print(doc.explain_processing())  # Complete processing explanation
"""

# Version information
__version__ = "1.0.0"
__author__ = "CortexOS Team"
__description__ = "Deterministic NLP Engine with Mathematical Certainty"
__spacy_version__ = "3.7.0"  # Emulated spaCy version for compatibility

# Core imports
from .api.cortex_nlp import CortexNLP
from .api.cortex_doc import Doc
from .api.cortex_token import Token
from .api.cortex_span import Span

# spaCy compatibility layer
from .api.spacy_compatibility import (
    load,
    blank,
    info,
    prefer_gpu,
    require_gpu,
    is_package,
    about,
    explain,
    registry,
    SpacyInterface
)

# Make the package behave like spaCy
from .api.spacy_compatibility import spacy_interface

# Export spaCy-compatible interface
def __getattr__(name):
    """
    Dynamic attribute access for spaCy compatibility.
    
    This allows the package to be used exactly like spaCy:
    import cortexos_nlp as spacy
    nlp = spacy.load("en_core_web_sm")
    """
    return getattr(spacy_interface, name)

# Core classes and functions available at package level
Language = CortexNLP

# Utility classes for spaCy compatibility
class util:
    """Utility functions (spaCy compatibility)."""
    from .api.spacy_compatibility import util_get_package_path as get_package_path

class displacy:
    """Visualization utilities (spaCy compatibility)."""
    from .api.spacy_compatibility import displacy_render as render

# Main exports
__all__ = [
    # Core classes
    'CortexNLP',
    'Doc', 
    'Token',
    'Span',
    'Language',
    
    # spaCy compatibility functions
    'load',
    'blank',
    'info',
    'prefer_gpu',
    'require_gpu',
    'is_package',
    'about',
    'explain',
    'registry',
    
    # Utility modules
    'util',
    'displacy',
    
    # Version info
    '__version__',
    '__author__',
    '__description__'
]

# Package metadata for spaCy compatibility
meta = {
    'name': 'cortexos_nlp',
    'version': __version__,
    'description': __description__,
    'author': __author__,
    'spacy_version': __spacy_version__,
    'deterministic': True,
    'mathematical_certainty': True
}

# Initialize message
def _show_welcome():
    """Show welcome message on first import."""
    import os
    if os.environ.get('CORTEXOS_QUIET') != '1':
        print(f"CortexOS NLP v{__version__} - Deterministic Language Processing")
        print("Mathematical certainty in every processing step.")
        print("Use 'import cortexos_nlp as spacy' for drop-in spaCy replacement.")

# Show welcome message (can be disabled with CORTEXOS_QUIET=1)
try:
    _show_welcome()
except:
    pass  # Silently fail if there are issues with output

