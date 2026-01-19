"""
VerdictVision Configuration
Central configuration for all model parameters and paths.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "preprocessed"
MODELS_DIR = BASE_DIR / "models"

# Input paths
RAW_DATA_DIR = DATA_DIR / "raw"
CASE_LAW_DIR = DATA_DIR / "case_law"

# Output paths
EMBEDDINGS_PATH = OUTPUT_DIR / "embeddings.npy"
CHUNKS_PATH = OUTPUT_DIR / "text_chunks.json"
METADATA_PATH = OUTPUT_DIR / "cases_metadata.csv"
FULL_DATA_PATH = OUTPUT_DIR / "cases_full.json"
CLASSIFICATION_PATH = OUTPUT_DIR / "classification_data.csv"

# ============================================================================
# DATA COLLECTION
# ============================================================================
CASE_LAW_BASE_URL = "https://static.case.law/cal-rptr-3d/{}/cases/"
LAST_VOLUME = 251
NUM_VOLUMES = 10
START_VOLUME = LAST_VOLUME - NUM_VOLUMES + 1

# Google Drive backup (fallback)
GDRIVE_FILE_ID = "1KvKVVkCfpxordzChDezxB3G3JNDdv6zh"

# ============================================================================
# TEXT PROCESSING
# ============================================================================
CHUNK_SIZE = 500  # words
CHUNK_OVERLAP = 100  # words

# ============================================================================
# EMBEDDING MODEL
# ============================================================================
EMBEDDING_MODEL = "joshcx/static-embedding-all-MiniLM-L6-v2"
EMBEDDING_DIM = 256
EMBEDDING_BATCH_SIZE = 32

# ============================================================================
# LLM MODEL
# ============================================================================
LLM_MODEL = "microsoft/phi-2"
LLM_MAX_NEW_TOKENS = 150
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9

# ============================================================================
# RETRIEVAL SYSTEM
# ============================================================================
# Hybrid search weights
SEMANTIC_WEIGHT = 0.4
TFIDF_WEIGHT = 0.3
METADATA_WEIGHT = 0.3

# Default retrieval parameters
DEFAULT_TOP_K = 5

# TF-IDF parameters
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)

# ============================================================================
# EVALUATION
# ============================================================================
EVAL_QUERIES = [
    {"query": "breach of contract damages California", "keywords": ["breach", "contract", "damages", "agreement"]},
    {"query": "negligence standard of care duty", "keywords": ["negligence", "duty", "care", "reasonable"]},
    {"query": "premises liability slip and fall injury", "keywords": ["premises", "liability", "slip", "fall", "injury"]},
    {"query": "fraud misrepresentation elements California", "keywords": ["fraud", "misrepresentation", "deceit", "reliance"]},
    {"query": "employment wrongful termination discharge", "keywords": ["employment", "wrongful", "termination", "discharge"]},
]

# Classification labels
VALID_OUTCOME_LABELS = ["affirmed", "reversed", "remanded"]

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# OPTIMIZATION SETTINGS
# ============================================================================
OPTIMIZATION_CONFIGS = [
    {"name": "Default (k=3, 150 tokens)", "top_k": 3, "max_new_tokens": 150},
    {"name": "Fast (k=2, 100 tokens)", "top_k": 2, "max_new_tokens": 100},
    {"name": "Very Fast (k=1, 80 tokens)", "top_k": 1, "max_new_tokens": 80},
]


def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [DATA_DIR, OUTPUT_DIR, MODELS_DIR, RAW_DATA_DIR, CASE_LAW_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
