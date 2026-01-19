"""
VerdictVision Embeddings Module
Creates and manages text embeddings for semantic search.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import time

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import (
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, CHUNKS_PATH, EMBEDDINGS_PATH,
    OUTPUT_DIR, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, ensure_directories
)


class EmbeddingManager:
    """Manages creation and storage of text embeddings."""
    
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.embedder: Optional[SentenceTransformer] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        ensure_directories()
    
    def load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        if self.embedder is None:
            print(f"Loading embedding model: {self.model_name}")
            self.embedder = SentenceTransformer(self.model_name)
            print(f"  Model loaded successfully!")
        return self.embedder
    
    def create_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings (for cosine similarity)
            
        Returns:
            Numpy array of embeddings
        """
        self.load_model()
        
        print(f"\nCreating embeddings for {len(texts)} texts...")
        start_time = time.time()
        
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=show_progress,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        elapsed = time.time() - start_time
        print(f"  Created {embeddings.shape[0]} embeddings in {elapsed:.2f}s")
        print(f"  Embedding dimensions: {embeddings.shape[1]}")
        
        return embeddings
    
    def create_tfidf_index(
        self,
        texts: List[str],
        max_features: int = TFIDF_MAX_FEATURES,
        ngram_range: tuple = TFIDF_NGRAM_RANGE
    ) -> TfidfVectorizer:
        """
        Create TF-IDF index for keyword search.
        
        Args:
            texts: List of text strings to index
            max_features: Maximum vocabulary size
            ngram_range: N-gram range (min_n, max_n)
            
        Returns:
            Fitted TfidfVectorizer
        """
        print(f"\nCreating TF-IDF index...")
        print(f"  Max features: {max_features}")
        print(f"  N-gram range: {ngram_range}")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        print(f"  TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"  Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        return self.tfidf_vectorizer
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        path: Optional[Path] = None
    ) -> Path:
        """Save embeddings to disk."""
        path = path or EMBEDDINGS_PATH
        np.save(path, embeddings)
        print(f"  Saved embeddings to: {path}")
        return path
    
    def load_embeddings(self, path: Optional[Path] = None) -> np.ndarray:
        """Load embeddings from disk."""
        path = path or EMBEDDINGS_PATH
        embeddings = np.load(path)
        print(f"  Loaded embeddings from: {path}")
        print(f"  Shape: {embeddings.shape}")
        return embeddings
    
    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Embed a single query text."""
        self.load_model()
        return self.embedder.encode(
            query,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
    
    def get_tfidf_query_vector(self, query: str):
        """Get TF-IDF vector for a query."""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not initialized. Call create_tfidf_index first.")
        return self.tfidf_vectorizer.transform([query])


class GloVeEmbeddings:
    """
    Alternative embeddings using GloVe static word vectors.
    Useful for two-stage retrieval systems.
    """
    
    def __init__(self, model_name: str = 'glove-wiki-gigaword-50'):
        self.model_name = model_name
        self.model = None
        self.dim = None
    
    def load_model(self, max_retries: int = 3):
        """Load GloVe model with retry logic."""
        import gensim.downloader as api
        
        for attempt in range(max_retries):
            try:
                print(f"  Attempt {attempt + 1}/{max_retries}: Loading '{self.model_name}'...")
                self.model = api.load(self.model_name)
                self.dim = self.model.vector_size
                print(f"  Successfully loaded! Dimensions: {self.dim}")
                return self.model
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {str(e)[:50]}...")
                if attempt < max_retries - 1:
                    print(f"  Retrying in 5 seconds...")
                    time.sleep(5)
        
        raise RuntimeError(f"Failed to load {self.model_name} after {max_retries} attempts")
    
    def get_document_embedding(self, text: str) -> np.ndarray:
        """
        Create document embedding by averaging word vectors.
        
        Args:
            text: Input text
            
        Returns:
            Document embedding vector
        """
        if self.model is None:
            self.load_model()
        
        words = text.lower().split()
        vectors = []
        
        for word in words:
            if word in self.model:
                vectors.append(self.model[word])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.dim)
    
    def embed_documents(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """Create embeddings for multiple documents."""
        if self.model is None:
            self.load_model()
        
        embeddings = []
        
        for i, text in enumerate(texts):
            embeddings.append(self.get_document_embedding(text))
            if show_progress and (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(texts)} documents...")
        
        return np.array(embeddings)


def create_embeddings_for_chunks(chunks_path: Path = CHUNKS_PATH) -> np.ndarray:
    """
    Convenience function to create embeddings for preprocessed chunks.
    
    Args:
        chunks_path: Path to chunks JSON file
        
    Returns:
        Embedding matrix
    """
    # Load chunks
    print("Loading chunks...")
    with open(chunks_path, 'r') as f:
        chunks = json.load(f)
    
    texts = [chunk['text'] for chunk in chunks]
    print(f"Loaded {len(texts)} chunks")
    
    # Create embeddings
    manager = EmbeddingManager()
    embeddings = manager.create_embeddings(texts)
    
    # Save embeddings
    manager.save_embeddings(embeddings)
    
    # Also create TF-IDF index
    manager.create_tfidf_index(texts)
    
    return embeddings


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create embeddings for VerdictVision")
    parser.add_argument("--chunks", type=str, help="Path to chunks JSON file")
    parser.add_argument("--output", type=str, help="Output path for embeddings")
    
    args = parser.parse_args()
    
    chunks_path = Path(args.chunks) if args.chunks else CHUNKS_PATH
    
    embeddings = create_embeddings_for_chunks(chunks_path)
    print(f"\nFinal embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
