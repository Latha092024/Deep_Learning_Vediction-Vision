"""
VerdictVision LLM Module
Handles loading and inference with the Phi-2 language model.
"""

import torch
from typing import List, Dict, Optional
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import LLM_MODEL, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE, LLM_TOP_P


class LLMManager:
    """Manages LLM loading and inference."""
    
    def __init__(
        self,
        model_name: str = LLM_MODEL,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        top_p: float = LLM_TOP_P
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self) -> None:
        """Load the LLM model and tokenizer."""
        if self.pipeline is not None:
            return
        
        print(f"Loading LLM: {self.model_name}")
        print(f"  Device: {self.device}")
        
        # Load tokenizer
        print("  Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Load model
        print("  Loading model (this may take 1-2 minutes)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        print("  LLM loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: bool = True
    ) -> str:
        """
        Generate text completion for a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Override default max tokens
            temperature: Override default temperature
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated text
        """
        self.load_model()
        
        output = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=do_sample,
            temperature=temperature or self.temperature,
            return_full_text=False
        )
        
        return output[0]["generated_text"].strip()
    
    def generate_with_context(
        self,
        question: str,
        context_cases: List[Dict],
        max_context_chars: int = 200
    ) -> str:
        """
        Generate answer using retrieved case context.
        
        Args:
            question: User question
            context_cases: List of retrieved case dictionaries
            max_context_chars: Max characters per case in context
            
        Returns:
            Generated answer
        """
        # Build prompt with context
        prompt = "You are a legal assistant. Answer based on these California cases.\n\n"
        
        for i, case in enumerate(context_cases, 1):
            prompt += f"Case {i}: {case['case_name']}\n"
            prompt += f"Court: {case.get('court', 'N/A')}\n"
            prompt += f"Key text: {case['text'][:max_context_chars]}...\n\n"
        
        prompt += f"Question: {question}\n\nAnswer (cite cases):"
        
        return self.generate(prompt)


class PromptTemplates:
    """Collection of prompt templates for different tasks."""
    
    @staticmethod
    def qa_prompt(question: str, cases: List[Dict], max_chars: int = 200) -> str:
        """Build Q&A prompt with case context."""
        prompt = "You are a legal assistant. Answer based on these California cases.\n\n"
        
        for i, case in enumerate(cases, 1):
            prompt += f"Case {i}: {case['case_name']}\n"
            prompt += f"Court: {case.get('court', 'N/A')}\n"
            prompt += f"Key text: {case['text'][:max_chars]}...\n\n"
        
        prompt += f"Question: {question}\n\nAnswer (cite cases):"
        return prompt
    
    @staticmethod
    def irac_prompt(question: str, cases: List[Dict], max_chars: int = 300) -> str:
        """Build IRAC analysis prompt."""
        prompt = """You are a legal analyst. Write an IRAC analysis based on these California cases.

IRAC Format:
- Issue: State the legal issue
- Rule: State the applicable legal rule
- Application: Apply the rule to the facts
- Conclusion: State your conclusion

Cases:
"""
        for i, case in enumerate(cases, 1):
            prompt += f"\n{i}. {case['case_name']}\n"
            prompt += f"   {case['text'][:max_chars]}...\n"
        
        prompt += f"\nQuestion: {question}\n\nIRAC Analysis:"
        return prompt
    
    @staticmethod
    def outcome_prediction_prompt(case_text: str, similar_cases: List[Dict]) -> str:
        """Build outcome prediction prompt."""
        prompt = """Based on similar California cases, predict the likely outcome of this case.

Similar Cases:
"""
        for i, case in enumerate(similar_cases, 1):
            outcome = case.get('outcome', 'unknown')
            prompt += f"{i}. {case['case_name']} - Outcome: {outcome}\n"
        
        prompt += f"\nCase to Predict:\n{case_text[:500]}...\n"
        prompt += "\nPredicted Outcome (affirmed/reversed/remanded):"
        return prompt


def main():
    """Demo LLM functionality."""
    print("="*70)
    print("VERDICTVISION LLM DEMO")
    print("="*70)
    
    # Initialize LLM
    llm = LLMManager()
    
    # Simple generation test
    test_prompt = "What are the elements of breach of contract in California?\n\nAnswer:"
    
    print(f"\nPrompt: {test_prompt}")
    print("\nGenerating response...")
    
    response = llm.generate(test_prompt, max_new_tokens=100)
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
