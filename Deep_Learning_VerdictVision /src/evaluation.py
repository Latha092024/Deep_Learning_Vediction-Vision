"""
VerdictVision Evaluation Module
Comprehensive evaluation suite for retrieval, Q&A, and outcome prediction.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_curve, auc, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import (
    EVAL_QUERIES, VALID_OUTCOME_LABELS, TEST_SIZE, RANDOM_STATE,
    CLASSIFICATION_PATH, OUTPUT_DIR, OPTIMIZATION_CONFIGS
)


class RetrievalEvaluator:
    """Evaluates retrieval system performance."""
    
    def __init__(self, retriever, chunks: List[Dict]):
        self.retriever = retriever
        self.chunks = chunks
    
    def precision_at_k(
        self,
        query: str,
        relevant_keywords: List[str],
        k: int = 5
    ) -> float:
        """Calculate Precision@K using keyword matching as relevance proxy."""
        results = self.retriever.hybrid_search(query, top_k=k)
        
        relevant_count = 0
        for result in results:
            text_lower = result['text'].lower()
            if any(kw.lower() in text_lower for kw in relevant_keywords):
                relevant_count += 1
        
        return relevant_count / k if k > 0 else 0.0
    
    def mean_reciprocal_rank(
        self,
        query: str,
        relevant_keywords: List[str],
        k: int = 10
    ) -> float:
        """Calculate Mean Reciprocal Rank."""
        results = self.retriever.hybrid_search(query, top_k=k)
        
        for i, result in enumerate(results, 1):
            text_lower = result['text'].lower()
            if any(kw.lower() in text_lower for kw in relevant_keywords):
                return 1.0 / i
        
        return 0.0
    
    def ndcg_at_k(
        self,
        query: str,
        relevant_keywords: List[str],
        k: int = 5
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        results = self.retriever.hybrid_search(query, top_k=k)
        
        relevances = []
        for result in results:
            text_lower = result['text'].lower()
            rel = 1 if any(kw.lower() in text_lower for kw in relevant_keywords) else 0
            relevances.append(rel)
        
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def measure_latency(self, query: str, k: int = 5, num_runs: int = 5) -> Dict[str, float]:
        """Measure retrieval latency."""
        latencies = []
        
        for _ in range(num_runs):
            start = time.perf_counter()
            self.retriever.hybrid_search(query, top_k=k)
            latencies.append((time.perf_counter() - start) * 1000)
        
        return {"avg_ms": np.mean(latencies), "p95_ms": np.percentile(latencies, 95)}
    
    def evaluate_all(self, queries: List[Dict] = EVAL_QUERIES) -> pd.DataFrame:
        """Run full evaluation on all queries."""
        print("=" * 80)
        print("RETRIEVAL SYSTEM EVALUATION")
        print("=" * 80)
        
        results = []
        
        for q_data in queries:
            query = q_data['query']
            keywords = q_data['keywords']
            
            print(f"\nEvaluating: {query[:50]}...")
            
            p5 = self.precision_at_k(query, keywords, k=5)
            p10 = self.precision_at_k(query, keywords, k=10)
            mrr = self.mean_reciprocal_rank(query, keywords)
            ndcg = self.ndcg_at_k(query, keywords)
            latency = self.measure_latency(query)
            
            results.append({
                "Query": query[:40] + "...",
                "P@5": p5, "P@10": p10, "MRR": mrr,
                "NDCG@5": ndcg, "Latency (ms)": latency["avg_ms"]
            })
        
        df = pd.DataFrame(results)
        
        print("\n" + "=" * 80)
        print("RETRIEVAL METRICS SUMMARY")
        print("=" * 80)
        print(f"Average P@5:      {df['P@5'].mean():.3f}")
        print(f"Average P@10:     {df['P@10'].mean():.3f}")
        print(f"Average MRR:      {df['MRR'].mean():.3f}")
        print(f"Average NDCG@5:   {df['NDCG@5'].mean():.3f}")
        print(f"Average Latency:  {df['Latency (ms)'].mean():.1f}ms")
        
        return df


class OutcomePredictionEvaluator:
    """Evaluates outcome prediction performance."""
    
    def __init__(self, classification_data_path: Path = CLASSIFICATION_PATH):
        self.data_path = classification_data_path
        self.clf_df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X_test_text = None
        self.tfidf = self.logreg = None
    
    def load_data(self) -> pd.DataFrame:
        """Load classification data."""
        self.clf_df = pd.read_csv(self.data_path)
        self.clf_df = self.clf_df[self.clf_df['outcome_label'].isin(VALID_OUTCOME_LABELS)].copy()
        
        print(f"Loaded {len(self.clf_df)} labelled cases")
        print("Label distribution:")
        print(self.clf_df['outcome_label'].value_counts(normalize=True).round(3))
        
        return self.clf_df
    
    def train_baseline(self) -> Tuple[float, str]:
        """Train TF-IDF + Logistic Regression baseline."""
        if self.clf_df is None:
            self.load_data()
        
        X_text = self.clf_df['full_text'].fillna("")
        y = self.clf_df['outcome_label']
        
        X_train_text, X_test_text, self.y_train, self.y_test = train_test_split(
            X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        self.X_test_text = X_test_text
        
        self.tfidf = TfidfVectorizer(max_features=5000)
        self.X_train = self.tfidf.fit_transform(X_train_text)
        self.X_test = self.tfidf.transform(X_test_text)
        
        self.logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        self.logreg.fit(self.X_train, self.y_train)
        
        y_pred = self.logreg.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        
        print("\nLogistic Regression Baseline:")
        print(f"Accuracy: {accuracy:.3f}")
        print(report)
        
        return accuracy, report
    
    def compute_majority_baseline(self) -> float:
        """Compute accuracy of always-predict-majority baseline."""
        if self.y_test is None:
            self.train_baseline()
        
        majority_class = Counter(self.y_train).most_common(1)[0][0]
        y_pred = [majority_class] * len(self.y_test)
        return accuracy_score(self.y_test, y_pred)
    
    def plot_confusion_matrix(
        self, y_true: List[str], y_pred: List[str],
        title: str = "Confusion Matrix", save_path: Optional[Path] = None
    ) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=VALID_OUTCOME_LABELS)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=VALID_OUTCOME_LABELS, yticklabels=VALID_OUTCOME_LABELS)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
    
    def plot_roc_curve(
        self, y_true: List[int], y_scores: List[float],
        title: str = "ROC Curve", save_path: Optional[Path] = None
    ) -> float:
        """Plot ROC curve for binary classification."""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
        
        return roc_auc


class LatencyOptimizer:
    """Measures and optimizes inference latency."""
    
    def __init__(self, qa_system):
        self.qa_system = qa_system
    
    def benchmark_settings(
        self, questions: List[str], settings: List[Dict] = OPTIMIZATION_CONFIGS
    ) -> pd.DataFrame:
        """Benchmark different configuration settings."""
        print("=" * 80)
        print("INFERENCE LATENCY OPTIMIZATION STUDY")
        print("=" * 80)
        
        results = []
        
        for config in settings:
            print(f"\nTesting: {config['name']}")
            latencies, lengths = [], []
            
            for q in questions:
                start = time.perf_counter()
                result = self.qa_system.query(q, mode="qa", top_k=config['top_k'])
                lat_ms = (time.perf_counter() - start) * 1000
                latencies.append(lat_ms)
                lengths.append(len(result['answer'].split()))
            
            results.append({
                "Setting": config['name'],
                "Avg Latency (ms)": np.mean(latencies),
                "P95 Latency (ms)": np.percentile(latencies, 95),
                "Avg Answer Length": np.mean(lengths)
            })
        
        df = pd.DataFrame(results)
        print("\n" + df.to_string(index=False))
        return df


def create_evaluation_report(
    retrieval_results: pd.DataFrame,
    outcome_results: Dict,
    latency_results: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR
) -> None:
    """Create comprehensive evaluation visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Retrieval metrics
    ax = axes[0, 0]
    metrics = ['P@5', 'P@10', 'MRR', 'NDCG@5']
    values = [retrieval_results[m].mean() for m in metrics]
    ax.bar(metrics, values, color='steelblue')
    ax.set_ylim(0, 1)
    ax.set_title('Retrieval Metrics')
    ax.set_ylabel('Score')
    
    # 2. Latency distribution
    ax = axes[0, 1]
    ax.hist(retrieval_results['Latency (ms)'], bins=10, color='coral', edgecolor='black')
    ax.set_title('Retrieval Latency Distribution')
    ax.set_xlabel('Latency (ms)')
    
    # 3. Outcome prediction comparison
    ax = axes[0, 2]
    methods = ['Majority', 'LogReg', 'LLM+RAG']
    accs = [
        outcome_results.get('majority_acc', 0),
        outcome_results.get('baseline_acc', 0),
        outcome_results.get('llm_acc', 0)
    ]
    ax.bar(methods, accs, color=['gray', 'steelblue', 'coral'])
    ax.set_ylim(0, 1)
    ax.set_title('Outcome Prediction Accuracy')
    
    # 4. Latency by setting
    ax = axes[1, 0]
    ax.barh(latency_results['Setting'], latency_results['Avg Latency (ms)'], color='teal')
    ax.set_title('Inference Latency by Setting')
    ax.set_xlabel('Latency (ms)')
    
    # 5. Speed vs Quality
    ax = axes[1, 1]
    ax.scatter(latency_results['Avg Latency (ms)'], latency_results['Avg Answer Length'], s=100, c='purple')
    ax.set_title('Speed vs Quality Tradeoff')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Answer Length')
    
    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')
    summary = f"""
    VerdictVision Summary
    =====================
    Dataset: 713 Cases
    
    Retrieval:
    - P@5: {retrieval_results['P@5'].mean():.3f}
    - MRR: {retrieval_results['MRR'].mean():.3f}
    
    Model: Phi-2 (2.7B)
    """
    ax.text(0.1, 0.5, summary, fontsize=10, family='monospace', va='center')
    
    plt.tight_layout()
    save_path = output_dir / 'evaluation_report.png'
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


def main():
    """Run evaluation demo."""
    print("=" * 80)
    print("VERDICTVISION EVALUATION MODULE")
    print("=" * 80)
    
    evaluator = OutcomePredictionEvaluator()
    try:
        evaluator.load_data()
        evaluator.train_baseline()
        print(f"\nMajority baseline: {evaluator.compute_majority_baseline():.3f}")
    except FileNotFoundError:
        print("\nRun preprocessing first to generate classification data.")


if __name__ == "__main__":
    main()
