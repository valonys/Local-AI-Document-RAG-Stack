"""Custom exceptions used across the Local RAG Stack."""


class LRSException(Exception):
    """Base exception."""


class ExtractionError(LRSException):
    """Raised when document extraction fails."""


class EmbeddingError(LRSException):
    """Raised when embedding generation fails."""


class StorageError(LRSException):
    """Raised when vector store operations fail."""


class RetrievalError(LRSException):
    """Raised when retrieval fails."""


class GenerationError(LRSException):
    """Raised when answer generation fails."""


class ConfigurationError(LRSException):
    """Raised when configuration is invalid."""
