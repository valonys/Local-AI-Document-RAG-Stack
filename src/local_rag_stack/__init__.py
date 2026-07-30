"""Local AI Document RAG Stack."""

__version__ = "0.1.0"

from .config import Settings, settings
from .pipeline import RAGPipeline

__all__ = ["RAGPipeline", "Settings", "settings", "__version__"]
