"""
VerdictVision - AI-Powered Legal Analytics
==========================================

A RAG-based system for analyzing California appellate cases using
hybrid retrieval and Microsoft Phi-2 LLM.

Modules:
    - data_collection: Download case law data
    - preprocessing: Text extraction and chunking
    - embeddings: Create semantic embeddings
    - retrieval: Hybrid search system
    - llm: LLM loading and inference
    - qa_system: Main Q&A application
    - evaluation: Performance evaluation
"""

from .data_collection import DataCollector
from .preprocessing import CasePreprocessor
from .embeddings import EmbeddingManager
from .retrieval import HybridRetriever
from .llm import LLMManager
from .qa_system import VerdictVisionQA
from .evaluation import RetrievalEvaluator, OutcomePredictionEvaluator

__version__ = "1.0.0"
__author__ = "VerdictVision Team"

__all__ = [
    "DataCollector",
    "CasePreprocessor", 
    "EmbeddingManager",
    "HybridRetriever",
    "LLMManager",
    "VerdictVisionQA",
    "RetrievalEvaluator",
    "OutcomePredictionEvaluator"
]
