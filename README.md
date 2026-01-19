# 🔍 VerdictVision: AI-Powered Legal Analytics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Retrieval-Augmented Generation (RAG) system for analyzing California appellate cases using hybrid retrieval and Microsoft Phi-2 LLM.

**CMPE 258 - Deep Learning Course Project | San Jose State University**

<p align="center">
  <img src="docs/architecture.png" alt="VerdictVision Architecture" width="800"/>
</p>

---

## 🎯 Overview

VerdictVision is an AI-powered legal analytics platform that enables:

- **🔍 Intelligent Case Search**: Find relevant cases using hybrid retrieval (semantic + keyword + metadata)
- **💬 Legal Q&A**: Get answers to legal questions with citations from actual cases
- **⚖️ IRAC Analysis**: Generate structured legal analysis (Issue, Rule, Application, Conclusion)
- **📊 Outcome Prediction**: Predict case outcomes based on similar historical cases

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid Retrieval** | Combines semantic embeddings (MiniLM), TF-IDF, and metadata scoring |
| **RAG Pipeline** | Retrieval-Augmented Generation using Microsoft Phi-2 (2.7B params) |
| **Legal Domain** | Trained on 713 California appellate cases |
| **Multiple Modes** | Q&A, Search, IRAC Analysis, Outcome Prediction |
| **Web Interface** | Interactive Gradio UI for easy interaction |

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Retrieval P@5** | 85.3% |
| **Retrieval MRR** | 0.82 |
| **Outcome Prediction Accuracy** | 85.3% |
| **P@5 Improvement (Hybrid vs Semantic-only)** | +22% |
| **Avg Query Latency** | < 3 seconds |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VerdictVision                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│   │  User   │───▶│  Query       │───▶│  Hybrid Retrieval   │   │
│   │  Query  │    │  Processing  │    │  ┌───────────────┐  │   │
│   └─────────┘    └──────────────┘    │  │ Semantic (40%)│  │   │
│                                       │  ├───────────────┤  │   │
│                                       │  │ TF-IDF  (30%) │  │   │
│                                       │  ├───────────────┤  │   │
│                                       │  │ Metadata(30%) │  │   │
│                                       │  └───────────────┘  │   │
│                                       └──────────┬──────────┘   │
│                                                  │              │
│                                                  ▼              │
│   ┌─────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│   │ Answer  │◀───│  Phi-2 LLM   │◀───│  Retrieved Cases    │   │
│   │         │    │  Generation  │    │  (Top-K Context)    │   │
│   └─────────┘    └──────────────┘    └─────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/VerdictVision.git
cd VerdictVision

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/VerdictVision/blob/main/notebooks/VerdictVision_Demo.ipynb)

### Basic Usage

```python
from src.qa_system import VerdictVisionQA

# Initialize system
qa = VerdictVisionQA()
qa.initialize()

# Ask a question
result = qa.query("What are the elements of breach of contract?", mode="qa")
print(result['answer'])

# Search for cases
results = qa.query("fraud misrepresentation California", mode="search")
for case in results['cases']:
    print(f"- {case['case_name']}")

# Predict outcome
result = qa.query(case_text, mode="predict")
print(f"Predicted: {result['predicted_outcome']}")
```

### Command Line Interface

```bash
# Download data
python main.py download --method gdrive

# Run preprocessing
python main.py preprocess

# Create embeddings
python main.py embeddings

# Launch web UI
python main.py serve --share

# Query from command line
python main.py query "What is negligence?" --mode qa
```

## 📁 Project Structure

```
VerdictVision/
├── configs/
│   ├── __init__.py
│   └── config.py              # Configuration parameters
├── src/
│   ├── __init__.py
│   ├── data_collection.py     # Download case law data
│   ├── preprocessing.py       # Text extraction & chunking
│   ├── embeddings.py          # Create semantic embeddings
│   ├── retrieval.py           # Hybrid search system
│   ├── llm.py                 # LLM loading & inference
│   ├── qa_system.py           # Main Q&A application
│   └── evaluation.py          # Performance evaluation
├── notebooks/
│   └── VerdictVision_Demo.ipynb  # Interactive demo
├── data/                      # Data directory (created at runtime)
│   ├── raw/
│   ├── case_law/
│   └── preprocessed/
├── main.py                    # CLI entry point
├── requirements.txt
├── LICENSE
└── README.md
```

## 🔧 Configuration

Key parameters in `configs/config.py`:

```python
# Embedding Model
EMBEDDING_MODEL = "joshcx/static-embedding-all-MiniLM-L6-v2"
EMBEDDING_DIM = 256

# LLM
LLM_MODEL = "microsoft/phi-2"
LLM_MAX_NEW_TOKENS = 150

# Retrieval Weights
SEMANTIC_WEIGHT = 0.4
TFIDF_WEIGHT = 0.3
METADATA_WEIGHT = 0.3

# Chunking
CHUNK_SIZE = 500  # words
CHUNK_OVERLAP = 100
```

## 📈 Evaluation Results

### Retrieval Performance

| Method | P@5 | P@10 | MRR | NDCG@5 |
|--------|-----|------|-----|--------|
| Semantic Only | 0.70 | 0.65 | 0.72 | 0.68 |
| TF-IDF Only | 0.60 | 0.55 | 0.65 | 0.58 |
| **Hybrid (Ours)** | **0.85** | **0.80** | **0.82** | **0.79** |

### Outcome Prediction

| Method | Accuracy | F1 (Weighted) |
|--------|----------|---------------|
| Majority Baseline | 65.2% | 0.52 |
| Logistic Regression | 78.4% | 0.76 |
| **LLM + RAG (Ours)** | **85.3%** | **0.84** |

### Latency Optimization

| Setting | Avg Latency | P95 Latency | Answer Length |
|---------|-------------|-------------|---------------|
| Default (k=3, 150 tokens) | 8,504ms | 12,679ms | 114 words |
| Fast (k=2, 100 tokens) | 3,213ms | 3,529ms | 74 words |
| Very Fast (k=1, 80 tokens) | 2,537ms | 2,907ms | 63 words |

## 🛠️ Technologies

- **Language Model**: Microsoft Phi-2 (2.7B parameters)
- **Embeddings**: MiniLM-L6-v2 (256 dimensions)
- **Vector Search**: Cosine similarity with numpy
- **Keyword Search**: Scikit-learn TF-IDF
- **Web Framework**: Gradio
- **Data Source**: [Case.law](https://case.law/) API

## 📚 Dataset

- **Source**: California Reporter, 3rd Series (Cal. Rptr. 3d)
- **Volumes**: 242-251 (most recent 10 volumes)
- **Cases**: 713 appellate court decisions
- **Chunks**: ~3,500 text segments (500 words each, 100 word overlap)

## 🎓 Course Information

- **Course**: CMPE 258 - Deep Learning
- **Institution**: San Jose State University
- **Semester**: Fall 2024

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Case.law](https://case.law/) for providing open access to legal data
- [Hugging Face](https://huggingface.co/) for transformer models
- Microsoft Research for the Phi-2 model
- SJSU CMPE 258 instructors and TAs

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

<p align="center">
  Made with ❤️ for CMPE 258 Deep Learning
</p>
