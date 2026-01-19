"""
VerdictVision Retrieval Module
Implements hybrid search combining semantic, TF-IDF, and metadata scoring.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import (
    CHUNKS_PATH, EMBEDDINGS_PATH, DEFAULT_TOP_K,
    SEMANTIC_WEIGHT, TFIDF_WEIGHT, METADATA_WEIGHT,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE
)
from src.embeddings import EmbeddingManager


class HybridRetriever:
    """
    Hybrid retrieval system combining semantic search, TF-IDF, and metadata scoring.
    
    The system uses a weighted fusion of three signals:
    1. Semantic similarity (dense embeddings)
    2. TF-IDF keyword matching (sparse)
    3. Metadata scoring (court, date, citations)
    """
    
    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        embeddings_path: Path = EMBEDDINGS_PATH,
        semantic_weight: float = SEMANTIC_WEIGHT,
        tfidf_weight: float = TFIDF_WEIGHT,
        metadata_weight: float = METADATA_WEIGHT
    ):
        self.chunks_path = chunks_path
        self.embeddings_path = embeddings_path
        
        # Weights for hybrid scoring
        self.semantic_weight = semantic_weight
        self.tfidf_weight = tfidf_weight
        self.metadata_weight = metadata_weight
        
        # Data (loaded lazily)
        self.chunks: Optional[List[Dict]] = None
        self.texts: Optional[List[str]] = None
        self.embeddings: Optional[np.ndarray] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        
        # Embedding manager
        self.embedding_manager = EmbeddingManager()
    
    def load_data(self) -> None:
        """Load chunks and embeddings from disk."""
        if self.chunks is None:
            print("Loading retrieval data...")
            
            # Load chunks
            with open(self.chunks_path, 'r') as f:
                self.chunks = json.load(f)
            self.texts = [chunk['text'] for chunk in self.chunks]
            print(f"  Loaded {len(self.chunks)} chunks")
            
            # Load embeddings
            self.embeddings = np.load(self.embeddings_path)
            print(f"  Loaded embeddings: {self.embeddings.shape}")
            
            # Create TF-IDF index
            self._build_tfidf_index()
    
    def _build_tfidf_index(self) -> None:
        """Build TF-IDF index for keyword search."""
        print("  Building TF-IDF index...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words='english'
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.texts)
        print(f"  TF-IDF matrix: {self.tfidf_matrix.shape}")
    
    def semantic_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Tuple[int, float]]:
        """
        Perform semantic search using dense embeddings.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (index, score) tuples
        """
        self.load_data()
        
        # Embed query
        query_embedding = self.embedding_manager.embed_query(query)
        
        # Compute similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(idx, similarities[idx]) for idx in top_indices]
    
    def tfidf_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Tuple[int, float]]:
        """
        Perform TF-IDF keyword search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (index, score) tuples
        """
        self.load_data()
        
        # Transform query
        query_vec = self.tfidf_vectorizer.transform([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(idx, similarities[idx]) for idx in top_indices]
    
    def compute_metadata_score(self, chunk: Dict, query: str) -> float:
        """
        Compute metadata relevance score.
        
        Considers:
        - Court level (Supreme Court > Appeals > Trial)
        - Recency (newer cases score higher)
        - Citation count (more citations = more authoritative)
        """
        score = 0.0
        
        # Court scoring
        court = chunk.get('court', '').lower()
        if 'supreme' in court:
            score += 0.3
        elif 'appeal' in court:
            score += 0.2
        else:
            score += 0.1
        
        # Recency scoring (if date available)
        date = chunk.get('decision_date', '')
        if date:
            try:
                year = int(date[:4])
                # Normalize year to 0-1 range (2000-2025)
                recency = (year - 2000) / 25
                score += max(0, min(0.3, recency * 0.3))
            except (ValueError, IndexError):
                pass
        
        # Citation scoring
        citations = chunk.get('citations', [])
        if isinstance(citations, list):
            citation_score = min(0.4, len(citations) * 0.1)
            score += citation_score
        
        return min(1.0, score)
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        semantic_weight: Optional[float] = None,
        tfidf_weight: Optional[float] = None,
        metadata_weight: Optional[float] = None
    ) -> List[Dict]:
        """
        Perform hybrid search combining all signals.
        
        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Override default semantic weight
            tfidf_weight: Override default TF-IDF weight
            metadata_weight: Override default metadata weight
            
        Returns:
            List of result dictionaries with scores
        """
        self.load_data()
        
        # Use provided weights or defaults
        w_sem = semantic_weight or self.semantic_weight
        w_tfidf = tfidf_weight or self.tfidf_weight
        w_meta = metadata_weight or self.metadata_weight
        
        # Normalize weights
        total_weight = w_sem + w_tfidf + w_meta
        w_sem /= total_weight
        w_tfidf /= total_weight
        w_meta /= total_weight
        
        # Get candidate pool (2x top_k from each method)
        candidate_k = min(top_k * 2, len(self.chunks))
        
        semantic_results = dict(self.semantic_search(query, candidate_k))
        tfidf_results = dict(self.tfidf_search(query, candidate_k))
        
        # Combine candidates
        all_candidates = set(semantic_results.keys()) | set(tfidf_results.keys())
        
        # Score all candidates
        scored_results = []
        
        for idx in all_candidates:
            chunk = self.chunks[idx]
            
            # Get individual scores
            sem_score = semantic_results.get(idx, 0.0)
            tfidf_score = tfidf_results.get(idx, 0.0)
            meta_score = self.compute_metadata_score(chunk, query)
            
            # Compute weighted final score
            final_score = (
                w_sem * sem_score +
                w_tfidf * tfidf_score +
                w_meta * meta_score
            )
            
            scored_results.append({
                **chunk,
                'scores': {
                    'semantic': sem_score,
                    'tfidf': tfidf_score,
                    'metadata': meta_score,
                    'final': final_score
                }
            })
        
        # Sort by final score and return top-k
        scored_results.sort(key=lambda x: x['scores']['final'], reverse=True)
        
        return scored_results[:top_k]
    
    def search_with_reranking(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        rerank_model: Optional[str] = None
    ) -> List[Dict]:
        """
        Two-stage retrieval with optional reranking.
        
        Stage 1: Fast hybrid retrieval
        Stage 2: (Optional) Neural reranking of top candidates
        """
        # Stage 1: Get candidates
        candidates = self.hybrid_search(query, top_k=top_k * 3)
        
        # Stage 2: Rerank if model provided
        if rerank_model is not None:
            # Placeholder for cross-encoder reranking
            # Would use something like: CrossEncoder(rerank_model)
            pass
        
        return candidates[:top_k]


class SimpleRetriever:
    """
    Simplified retriever for evaluation purposes.
    Uses only semantic search without hybrid scoring.
    """
    
    def __init__(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
        embedding_manager: EmbeddingManager
    ):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedding_manager = embedding_manager
    
    def search(self, query: str, k: int = DEFAULT_TOP_K) -> List[Dict]:
        """Simple semantic search."""
        query_embedding = self.embedding_manager.embed_query(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            result = self.chunks[idx].copy()
            result['score'] = float(similarities[idx])
            results.append(result)
        
        return results


def hybrid_search_eval(
    query: str,
    chunks: List[Dict],
    embeddings: np.ndarray,
    tfidf_matrix,
    tfidf_vectorizer: TfidfVectorizer,
    embedding_manager: EmbeddingManager,
    k: int = DEFAULT_TOP_K
) -> List[Dict]:
    """
    Standalone hybrid search function for evaluation.
    
    This function is designed to be used in evaluation loops where
    you don't want to instantiate the full HybridRetriever class.
    """
    # Semantic search
    query_embedding = embedding_manager.embed_query(query)
    semantic_sims = cosine_similarity([query_embedding], embeddings)[0]
    
    # TF-IDF search
    query_vec = tfidf_vectorizer.transform([query])
    tfidf_sims = cosine_similarity(query_vec, tfidf_matrix)[0]
    
    # Combine scores
    combined_scores = (
        SEMANTIC_WEIGHT * semantic_sims +
        TFIDF_WEIGHT * tfidf_sims
    )
    
    # Get top-k
    top_indices = np.argsort(combined_scores)[::-1][:k]
    
    results = []
    for idx in top_indices:
        result = chunks[idx].copy()
        result['score'] = float(combined_scores[idx])
        results.append(result)
    
    return results


def main():
    """Demonstrate retrieval functionality."""
    print("="*70)
    print("VERDICTVISION RETRIEVAL SYSTEM DEMO")
    print("="*70)
    
    # Initialize retriever
    retriever = HybridRetriever()
    
    # Test queries
    test_queries = [
        "breach of contract damages California",
        "negligence standard of care",
        "fraud elements misrepresentation"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        results = retriever.hybrid_search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result['case_name']}")
            print(f"    Court: {result.get('court', 'N/A')}")
            print(f"    Score: {result['scores']['final']:.4f}")
            print(f"    Text: {result['text'][:200]}...")


if __name__ == "__main__":
    main()
