#!/usr/bin/env python3
"""
VerdictVision - Main Entry Point
================================

Run the complete VerdictVision pipeline or individual components.

Usage:
    python main.py --help                    # Show help
    python main.py preprocess               # Run preprocessing
    python main.py embeddings               # Create embeddings
    python main.py evaluate                 # Run evaluation
    python main.py serve                    # Launch Gradio UI
    python main.py query "your question"    # Query the system
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from configs.config import ensure_directories


def cmd_preprocess(args):
    """Run preprocessing pipeline."""
    from src.preprocessing import CasePreprocessor
    
    preprocessor = CasePreprocessor(
        input_dir=Path(args.input) if args.input else None,
        output_dir=Path(args.output) if args.output else None
    )
    preprocessor.run_pipeline()


def cmd_embeddings(args):
    """Create embeddings for chunks."""
    from src.embeddings import create_embeddings_for_chunks
    from configs.config import CHUNKS_PATH
    
    chunks_path = Path(args.chunks) if args.chunks else CHUNKS_PATH
    create_embeddings_for_chunks(chunks_path)


def cmd_evaluate(args):
    """Run evaluation suite."""
    from src.evaluation import OutcomePredictionEvaluator
    
    evaluator = OutcomePredictionEvaluator()
    evaluator.load_data()
    evaluator.train_baseline()
    print(f"\nMajority baseline: {evaluator.compute_majority_baseline():.3f}")


def cmd_serve(args):
    """Launch Gradio web interface."""
    from src.qa_system import VerdictVisionQA, create_gradio_interface
    
    print("Initializing VerdictVision...")
    qa_system = VerdictVisionQA()
    qa_system.initialize()
    
    print("\nLaunching Gradio interface...")
    interface = create_gradio_interface(qa_system)
    interface.launch(share=args.share, server_port=args.port)


def cmd_query(args):
    """Query the system."""
    from src.qa_system import VerdictVisionQA
    
    qa_system = VerdictVisionQA()
    qa_system.initialize()
    
    result = qa_system.query(args.question, mode=args.mode)
    
    print("\n" + "=" * 70)
    print(f"Mode: {result['mode']}")
    print("=" * 70)
    
    if args.mode == "qa":
        print(f"\nQuestion: {args.question}")
        print(f"\nAnswer:\n{result['answer']}")
        print("\nSources:")
        for i, case in enumerate(result['cases'][:3], 1):
            print(f"  {i}. {case['case_name']}")
    
    elif args.mode == "search":
        print(f"\nQuery: {args.question}")
        for i, case in enumerate(result['cases'], 1):
            print(f"\n{i}. {case['case_name']}")
            print(f"   Score: {case['scores']['final']:.4f}")
            print(f"   Preview: {case['text'][:150]}...")
    
    elif args.mode == "predict":
        print(f"\nPredicted Outcome: {result['predicted_outcome'].upper()}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
    
    print(f"\nLatency: {result['latency_ms']:.1f}ms")


def cmd_download(args):
    """Download case law data."""
    from src.data_collection import DataCollector
    
    collector = DataCollector()
    
    if args.method == "api":
        collector.download_from_case_law()
    else:
        collector.download_from_gdrive()


def main():
    parser = argparse.ArgumentParser(
        description="VerdictVision - AI-Powered Legal Analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download command
    dl_parser = subparsers.add_parser("download", help="Download case law data")
    dl_parser.add_argument("--method", choices=["api", "gdrive"], default="gdrive")
    
    # Preprocess command
    pre_parser = subparsers.add_parser("preprocess", help="Run preprocessing pipeline")
    pre_parser.add_argument("--input", help="Input directory with JSON files")
    pre_parser.add_argument("--output", help="Output directory for preprocessed data")
    
    # Embeddings command
    emb_parser = subparsers.add_parser("embeddings", help="Create embeddings")
    emb_parser.add_argument("--chunks", help="Path to chunks JSON file")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation suite")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Launch Gradio web interface")
    serve_parser.add_argument("--share", action="store_true", help="Create public link")
    serve_parser.add_argument("--port", type=int, default=7860, help="Server port")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("question", help="Question or search query")
    query_parser.add_argument("--mode", choices=["qa", "search", "irac", "predict"],
                             default="qa", help="Query mode")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    ensure_directories()
    
    commands = {
        "download": cmd_download,
        "preprocess": cmd_preprocess,
        "embeddings": cmd_embeddings,
        "evaluate": cmd_evaluate,
        "serve": cmd_serve,
        "query": cmd_query
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
