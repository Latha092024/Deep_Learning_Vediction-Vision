"""
VerdictVision Q&A System
Main application module integrating retrieval and LLM for legal Q&A.
"""

import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import Counter

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import DEFAULT_TOP_K, LLM_MAX_NEW_TOKENS
from src.retrieval import HybridRetriever
from src.llm import LLMManager, PromptTemplates


class VerdictVisionQA:
    """
    Main Q&A system combining retrieval and LLM generation.
    
    Features:
    - Hybrid retrieval (semantic + TF-IDF + metadata)
    - RAG-based question answering
    - IRAC legal analysis generation
    - Outcome prediction
    """
    
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        llm: Optional[LLMManager] = None
    ):
        self.retriever = retriever or HybridRetriever()
        self.llm = llm or LLMManager()
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return
        
        print("="*70)
        print("INITIALIZING VERDICTVISION Q&A SYSTEM")
        print("="*70)
        
        # Load retrieval data
        print("\n[1/2] Loading retrieval system...")
        self.retriever.load_data()
        
        # Load LLM
        print("\n[2/2] Loading LLM...")
        self.llm.load_model()
        
        self._initialized = True
        print("\nSystem ready!")
    
    def retrieve_cases(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict]:
        """
        Retrieve relevant cases for a query.
        
        Args:
            query: User question or search query
            top_k: Number of cases to retrieve
            
        Returns:
            List of relevant case chunks
        """
        self.initialize()
        return self.retriever.hybrid_search(query, top_k=top_k)
    
    def answer_question(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS
    ) -> Tuple[str, List[Dict]]:
        """
        Answer a legal question using RAG.
        
        Args:
            question: User question
            top_k: Number of cases to retrieve
            max_new_tokens: Max tokens for response
            
        Returns:
            Tuple of (answer, retrieved_cases)
        """
        self.initialize()
        
        # Retrieve relevant cases
        cases = self.retrieve_cases(question, top_k=top_k)
        
        # Build prompt
        prompt = PromptTemplates.qa_prompt(question, cases)
        
        # Generate answer
        answer = self.llm.generate(prompt, max_new_tokens=max_new_tokens)
        
        return answer, cases
    
    def generate_irac(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K
    ) -> Tuple[str, List[Dict]]:
        """
        Generate IRAC legal analysis.
        
        Args:
            question: Legal question to analyze
            top_k: Number of cases to retrieve
            
        Returns:
            Tuple of (IRAC analysis, retrieved_cases)
        """
        self.initialize()
        
        # Retrieve cases
        cases = self.retrieve_cases(question, top_k=top_k)
        
        # Build IRAC prompt
        prompt = PromptTemplates.irac_prompt(question, cases)
        
        # Generate analysis
        analysis = self.llm.generate(prompt, max_new_tokens=300)
        
        return analysis, cases
    
    def predict_outcome(
        self,
        case_text: str,
        k: int = 5
    ) -> Tuple[str, float, List[Dict]]:
        """
        Predict case outcome based on similar cases.
        
        Uses a heuristic approach: majority vote from similar cases.
        
        Args:
            case_text: Text of the case to predict
            k: Number of similar cases to consider
            
        Returns:
            Tuple of (predicted_outcome, confidence, similar_cases)
        """
        self.initialize()
        
        # Find similar cases
        similar_cases = self.retrieve_cases(case_text, top_k=k)
        
        # Count outcomes
        outcomes = [case.get('outcome', '').lower() for case in similar_cases]
        outcomes = [o for o in outcomes if o in ['affirmed', 'reversed', 'remanded']]
        
        if not outcomes:
            return 'affirmed', 0.0, similar_cases  # Default
        
        # Majority vote
        outcome_counts = Counter(outcomes)
        predicted_outcome, count = outcome_counts.most_common(1)[0]
        confidence = count / len(outcomes)
        
        return predicted_outcome, confidence, similar_cases
    
    def query(
        self,
        question: str,
        mode: str = "qa",
        top_k: int = DEFAULT_TOP_K
    ) -> Dict:
        """
        Unified query interface.
        
        Args:
            question: User question
            mode: One of "qa", "irac", "predict", "search"
            top_k: Number of cases to consider
            
        Returns:
            Dictionary with results based on mode
        """
        self.initialize()
        
        start_time = time.perf_counter()
        
        if mode == "qa":
            answer, cases = self.answer_question(question, top_k=top_k)
            result = {
                "mode": "qa",
                "question": question,
                "answer": answer,
                "cases": cases
            }
        
        elif mode == "irac":
            analysis, cases = self.generate_irac(question, top_k=top_k)
            result = {
                "mode": "irac",
                "question": question,
                "analysis": analysis,
                "cases": cases
            }
        
        elif mode == "predict":
            outcome, confidence, cases = self.predict_outcome(question, k=top_k)
            result = {
                "mode": "predict",
                "case_text": question[:500],
                "predicted_outcome": outcome,
                "confidence": confidence,
                "similar_cases": cases
            }
        
        elif mode == "search":
            cases = self.retrieve_cases(question, top_k=top_k)
            result = {
                "mode": "search",
                "query": question,
                "cases": cases
            }
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'qa', 'irac', 'predict', or 'search'")
        
        result["latency_ms"] = (time.perf_counter() - start_time) * 1000
        
        return result


def create_gradio_interface(qa_system: VerdictVisionQA):
    """
    Create Gradio web interface for VerdictVision.
    
    Args:
        qa_system: Initialized VerdictVisionQA instance
        
    Returns:
        Gradio Blocks interface
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is required. Install with: pip install gradio")
    
    def qa_fn(question: str) -> str:
        if not question.strip():
            return "Please enter a question."
        result = qa_system.query(question, mode="qa")
        
        output = f"**Answer:**\n{result['answer']}\n\n"
        output += "**Sources:**\n"
        for i, case in enumerate(result['cases'][:3], 1):
            output += f"{i}. {case['case_name']}\n"
        
        return output
    
    def irac_fn(question: str) -> str:
        if not question.strip():
            return "Please enter a legal question."
        result = qa_system.query(question, mode="irac")
        return result['analysis']
    
    def predict_fn(case_text: str) -> str:
        if not case_text.strip():
            return "Please enter case text."
        result = qa_system.query(case_text, mode="predict")
        
        output = f"**Predicted Outcome:** {result['predicted_outcome'].upper()}\n"
        output += f"**Confidence:** {result['confidence']*100:.1f}%\n\n"
        output += "**Similar Cases:**\n"
        for i, case in enumerate(result['similar_cases'][:3], 1):
            output += f"{i}. {case['case_name']} ({case.get('outcome', 'N/A')})\n"
        
        return output
    
    def search_fn(query: str) -> str:
        if not query.strip():
            return "Please enter a search query."
        result = qa_system.query(query, mode="search")
        
        output = ""
        for i, case in enumerate(result['cases'], 1):
            output += f"**{i}. {case['case_name']}**\n"
            output += f"Court: {case.get('court', 'N/A')}\n"
            output += f"Date: {case.get('decision_date', 'N/A')}\n"
            output += f"Score: {case['scores']['final']:.4f}\n"
            output += f"Preview: {case['text'][:200]}...\n\n"
        
        return output
    
    with gr.Blocks(title="VerdictVision") as interface:
        gr.Markdown("# VerdictVision - AI-Powered Legal Analytics")
        gr.Markdown("Analyze California appellate cases using RAG + Phi-2 LLM")
        
        with gr.Tabs():
            with gr.Tab("Q&A"):
                qa_input = gr.Textbox(
                    label="Ask a legal question",
                    placeholder="What are the elements of breach of contract in California?"
                )
                qa_output = gr.Markdown(label="Answer")
                qa_btn = gr.Button("Ask")
                qa_btn.click(qa_fn, qa_input, qa_output)
            
            with gr.Tab("IRAC Analysis"):
                irac_input = gr.Textbox(
                    label="Enter legal question for IRAC analysis",
                    placeholder="What is the standard for negligence in slip and fall cases?"
                )
                irac_output = gr.Markdown(label="IRAC Analysis")
                irac_btn = gr.Button("Generate IRAC")
                irac_btn.click(irac_fn, irac_input, irac_output)
            
            with gr.Tab("Outcome Prediction"):
                predict_input = gr.Textbox(
                    label="Enter case text",
                    placeholder="Paste case summary or key facts...",
                    lines=5
                )
                predict_output = gr.Markdown(label="Prediction")
                predict_btn = gr.Button("Predict Outcome")
                predict_btn.click(predict_fn, predict_input, predict_output)
            
            with gr.Tab("Case Search"):
                search_input = gr.Textbox(
                    label="Search query",
                    placeholder="fraud misrepresentation California"
                )
                search_output = gr.Markdown(label="Results")
                search_btn = gr.Button("Search")
                search_btn.click(search_fn, search_input, search_output)
    
    return interface


def main():
    """Run VerdictVision demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VerdictVision Q&A System")
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio interface")
    parser.add_argument("--question", type=str, help="Question to answer")
    parser.add_argument("--mode", type=str, default="qa", 
                       choices=["qa", "irac", "predict", "search"])
    
    args = parser.parse_args()
    
    # Initialize system
    qa_system = VerdictVisionQA()
    qa_system.initialize()
    
    if args.gradio:
        interface = create_gradio_interface(qa_system)
        interface.launch(share=True)
    elif args.question:
        result = qa_system.query(args.question, mode=args.mode)
        print(f"\nResult:\n{result}")
    else:
        # Interactive demo
        print("\n" + "="*70)
        print("VERDICTVISION INTERACTIVE DEMO")
        print("="*70)
        
        test_questions = [
            "What are the elements of breach of contract in California?",
            "How do courts determine negligence in personal injury cases?",
        ]
        
        for q in test_questions:
            print(f"\nQuestion: {q}")
            result = qa_system.query(q, mode="qa")
            print(f"Answer: {result['answer'][:500]}...")
            print(f"Latency: {result['latency_ms']:.1f}ms")


if __name__ == "__main__":
    main()
