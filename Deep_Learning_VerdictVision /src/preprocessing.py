"""
VerdictVision Preprocessing Module
Handles text extraction, cleaning, chunking, and metadata extraction from case law.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import (
    CASE_LAW_DIR, OUTPUT_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    METADATA_PATH, FULL_DATA_PATH, CHUNKS_PATH, CLASSIFICATION_PATH,
    VALID_OUTCOME_LABELS, ensure_directories
)


class CasePreprocessor:
    """Handles preprocessing of case law documents."""
    
    def __init__(self, input_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        self.input_dir = input_dir or CASE_LAW_DIR
        self.output_dir = output_dir or OUTPUT_DIR
        ensure_directories()
    
    def load_cases(self) -> List[Dict]:
        """
        Load and validate JSON case files.
        
        Returns:
            List of valid case dictionaries
        """
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Directory '{self.input_dir}' not found!")
        
        json_files = [f for f in os.listdir(self.input_dir) if f.endswith('.json')]
        print(f"\nInput directory: {self.input_dir}")
        print(f"Found {len(json_files)} JSON files")
        
        if len(json_files) == 0:
            return []
        
        cases = []
        errors = []
        
        print(f"\nLoading cases...")
        
        for i, filename in enumerate(json_files, 1):
            filepath = os.path.join(self.input_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    case = json.load(f)
                    
                    if 'id' in case and 'casebody' in case:
                        cases.append(case)
                        if i % 100 == 0:
                            print(f"  Loaded {i}/{len(json_files)} files...")
                    else:
                        errors.append(f"{filename}: Missing required fields")
            
            except json.JSONDecodeError as e:
                errors.append(f"{filename}: Invalid JSON - {e}")
            except Exception as e:
                errors.append(f"{filename}: {e}")
        
        print(f"\n{'='*70}")
        print("LOADING SUMMARY")
        print(f"{'='*70}")
        print(f"Successfully loaded: {len(cases)} cases")
        
        if errors:
            print(f"Errors: {len(errors)}")
            print("\nFirst 3 errors:")
            for err in errors[:3]:
                print(f"  - {err}")
        
        return cases
    
    @staticmethod
    def extract_metadata(case: Dict) -> Dict:
        """Extract structured metadata from a case."""
        metadata = {
            'case_id': case.get('id', ''),
            'case_name': case.get('name', ''),
            'name_abbreviation': case.get('name_abbreviation', ''),
            'decision_date': case.get('decision_date', ''),
            'docket_number': case.get('docket_number', ''),
        }
        
        # Court
        court = case.get('court', {})
        metadata['court_name'] = court.get('name', '') if isinstance(court, dict) else str(court)
        
        # Jurisdiction
        jurisdiction = case.get('jurisdiction', {})
        metadata['jurisdiction'] = jurisdiction.get('name', '') if isinstance(jurisdiction, dict) else str(jurisdiction)
        
        # Citations
        citations = []
        if 'citations' in case:
            for cite in case['citations']:
                if isinstance(cite, dict) and 'cite' in cite:
                    citations.append(cite['cite'])
                elif isinstance(cite, str):
                    citations.append(cite)
        metadata['citations'] = citations
        
        # Judges from casebody
        judges = []
        if 'casebody' in case and 'judges' in case['casebody']:
            judges = case['casebody']['judges']
            if not isinstance(judges, list):
                judges = [judges]
        metadata['judges'] = judges
        
        return metadata
    
    @staticmethod
    def extract_full_text(case: Dict) -> str:
        """Extract complete case text from JSON structure."""
        text_parts = []
        
        # Case name
        if 'name' in case:
            text_parts.append(f"CASE: {case['name']}")
        
        # Citations
        if 'citations' in case:
            cites = [c.get('cite', '') if isinstance(c, dict) else str(c)
                    for c in case['citations']]
            if cites:
                text_parts.append(f"CITATIONS: {', '.join(cites)}")
        
        # Casebody content
        if 'casebody' in case:
            casebody = case['casebody']
            
            # Head matter
            if 'head_matter' in casebody:
                text_parts.append(f"SUMMARY: {casebody['head_matter']}")
            
            # Opinions
            if 'opinions' in casebody:
                for opinion in casebody['opinions']:
                    if isinstance(opinion, dict):
                        if 'type' in opinion:
                            text_parts.append(f"\n[{opinion['type'].upper()}]")
                        if 'text' in opinion:
                            text_parts.append(opinion['text'])
                        if 'author' in opinion:
                            text_parts.append(f"Author: {opinion['author']}")
        
        return '\n\n'.join(text_parts)
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep legal formatting
        text = re.sub(r'[^\w\s.,;:()\'\"§\-–—]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    @staticmethod
    def extract_outcome(text: str, case_name: str = "") -> str:
        """
        Extract case outcome (affirmed/reversed/remanded) from text.
        
        Uses pattern matching on common legal outcome phrases.
        """
        text_lower = text.lower()
        
        outcome_patterns = {
            'affirmed': [
                r'judgment\s+(?:is\s+)?affirmed',
                r'order\s+(?:is\s+)?affirmed',
                r'we\s+affirm',
                r'is\s+affirmed',
                r'affirmed\s+in\s+(?:part|full)'
            ],
            'reversed': [
                r'judgment\s+(?:is\s+)?reversed',
                r'order\s+(?:is\s+)?reversed',
                r'we\s+reverse',
                r'is\s+reversed',
                r'reversed\s+and\s+remanded'
            ],
            'remanded': [
                r'remanded\s+for',
                r'case\s+(?:is\s+)?remanded',
                r'we\s+remand',
                r'remanded\s+to'
            ]
        }
        
        for outcome, patterns in outcome_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return outcome
        
        return 'unknown'
    
    def process_cases(self, cases: List[Dict]) -> pd.DataFrame:
        """
        Process all cases into a structured DataFrame.
        
        Args:
            cases: List of raw case dictionaries
            
        Returns:
            DataFrame with processed case data
        """
        print(f"\n{'='*70}")
        print("PROCESSING CASES")
        print(f"{'='*70}")
        
        processed_data = []
        
        for i, case in enumerate(cases, 1):
            # Extract metadata
            metadata = self.extract_metadata(case)
            
            # Extract and clean text
            full_text = self.extract_full_text(case)
            clean_text = self.clean_text(full_text)
            
            # Extract outcome
            outcome = self.extract_outcome(full_text, metadata.get('case_name', ''))
            
            # Combine all data
            processed_case = {
                **metadata,
                'full_text': clean_text,
                'text_length': len(clean_text),
                'word_count': len(clean_text.split()),
                'outcome_label': outcome,
                'num_citations': len(metadata['citations']),
                'num_judges': len(metadata['judges'])
            }
            processed_data.append(processed_case)
            
            if i % 100 == 0:
                print(f"  Processed {i}/{len(cases)} cases...")
        
        df = pd.DataFrame(processed_data)
        
        print(f"\n{'='*70}")
        print("PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Total cases processed: {len(df)}")
        print(f"Average text length: {df['word_count'].mean():.0f} words")
        print(f"\nOutcome distribution:")
        print(df['outcome_label'].value_counts())
        
        return df
    
    @staticmethod
    def create_text_chunks(
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP
    ) -> List[Dict]:
        """
        Create overlapping text chunks for RAG retrieval.
        
        Args:
            text: Full document text
            chunk_size: Number of words per chunk
            overlap: Number of overlapping words between chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or len(text) < 100:
            return [{'text': text, 'chunk_id': 0, 'length': len(text.split())}]
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'start_idx': i,
                'length': len(chunk_words)
            })
            
            if len(chunk_words) < chunk_size:
                break
        
        return chunks if chunks else [{'text': text, 'chunk_id': 0, 'length': len(text.split())}]
    
    def create_all_chunks(self, df: pd.DataFrame) -> List[Dict]:
        """
        Create text chunks for all cases in the DataFrame.
        
        Args:
            df: DataFrame with processed case data
            
        Returns:
            List of all chunks with case metadata
        """
        print(f"\n{'='*70}")
        print("CREATING TEXT CHUNKS")
        print(f"{'='*70}")
        print(f"Chunk size: {CHUNK_SIZE} words")
        print(f"Overlap: {CHUNK_OVERLAP} words")
        
        all_chunks = []
        
        for idx, row in df.iterrows():
            chunks = self.create_text_chunks(row['full_text'])
            
            for chunk in chunks:
                all_chunks.append({
                    'case_id': row['case_id'],
                    'case_name': row['case_name'],
                    'court': row['court_name'],
                    'decision_date': row['decision_date'],
                    'citations': row['citations'],
                    'outcome': row['outcome_label'],
                    **chunk
                })
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        print(f"Average chunks per case: {len(all_chunks) / len(df):.1f}")
        
        return all_chunks
    
    def save_preprocessed_data(
        self,
        df: pd.DataFrame,
        chunks: List[Dict]
    ) -> Dict[str, Path]:
        """
        Save all preprocessed data to disk.
        
        Args:
            df: Processed cases DataFrame
            chunks: List of text chunks
            
        Returns:
            Dictionary of saved file paths
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("SAVING PREPROCESSED DATA")
        print(f"{'='*70}")
        print(f"Output directory: {self.output_dir}")
        
        saved_files = {}
        
        # 1. Metadata CSV
        print("\n[1] Saving metadata CSV")
        metadata_df = df[['case_id', 'case_name', 'court_name', 'decision_date',
                         'num_citations', 'num_judges', 'outcome_label',
                         'text_length', 'word_count']].copy()
        metadata_df.to_csv(METADATA_PATH, index=False)
        saved_files['metadata'] = METADATA_PATH
        print(f"    Saved: {METADATA_PATH}")
        
        # 2. Full case data JSON
        print("[2] Saving full case data JSON")
        df_json = df.to_dict('records')
        with open(FULL_DATA_PATH, 'w') as f:
            json.dump(df_json, f, indent=2)
        saved_files['full_data'] = FULL_DATA_PATH
        print(f"    Saved: {FULL_DATA_PATH}")
        
        # 3. Text chunks JSON
        print("[3] Saving text chunks JSON")
        with open(CHUNKS_PATH, 'w') as f:
            json.dump(chunks, f, indent=2)
        saved_files['chunks'] = CHUNKS_PATH
        print(f"    Saved: {CHUNKS_PATH}")
        
        # 4. Classification data CSV
        print("[4] Saving classification data CSV")
        valid_df = df[df['outcome_label'].isin(VALID_OUTCOME_LABELS)]
        class_df = valid_df[['case_id', 'case_name', 'full_text', 'outcome_label']].copy()
        class_df.to_csv(CLASSIFICATION_PATH, index=False)
        saved_files['classification'] = CLASSIFICATION_PATH
        print(f"    Saved: {CLASSIFICATION_PATH} ({len(class_df)} cases)")
        
        return saved_files
    
    def run_pipeline(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Run the complete preprocessing pipeline.
        
        Returns:
            Tuple of (processed DataFrame, chunks list)
        """
        print("="*70)
        print("VERDICTVISION PREPROCESSING PIPELINE")
        print("="*70)
        
        # Load cases
        cases = self.load_cases()
        if not cases:
            raise ValueError("No valid cases loaded!")
        
        # Process cases
        df = self.process_cases(cases)
        
        # Create chunks
        chunks = self.create_all_chunks(df)
        
        # Save data
        self.save_preprocessed_data(df, chunks)
        
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE!")
        print(f"{'='*70}")
        
        return df, chunks


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess VerdictVision case law data")
    parser.add_argument("--input", type=str, help="Input directory with JSON files")
    parser.add_argument("--output", type=str, help="Output directory for preprocessed data")
    
    args = parser.parse_args()
    
    preprocessor = CasePreprocessor(
        input_dir=Path(args.input) if args.input else None,
        output_dir=Path(args.output) if args.output else None
    )
    
    preprocessor.run_pipeline()


if __name__ == "__main__":
    main()
