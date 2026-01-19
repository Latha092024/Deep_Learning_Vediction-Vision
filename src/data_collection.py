"""
VerdictVision Data Collection Module
Downloads California case law data from Case.law API or Google Drive backup.
"""

import os
import requests
import zipfile
from pathlib import Path
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from typing import Optional

try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import (
    CASE_LAW_BASE_URL, START_VOLUME, LAST_VOLUME,
    GDRIVE_FILE_ID, CASE_LAW_DIR, RAW_DATA_DIR, ensure_directories
)


class DataCollector:
    """Handles downloading and extracting case law data."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or RAW_DATA_DIR
        self.headers = {"User-Agent": "Mozilla/5.0"}
        ensure_directories()
    
    def _get_soup(self, url: str) -> BeautifulSoup:
        """Fetch and parse HTML from URL."""
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    
    def download_from_case_law(
        self,
        start_volume: int = START_VOLUME,
        end_volume: int = LAST_VOLUME
    ) -> int:
        """
        Download case law JSON files directly from Case.law API.
        
        Args:
            start_volume: Starting volume number
            end_volume: Ending volume number (inclusive)
            
        Returns:
            Number of files downloaded
        """
        os.makedirs(self.output_dir, exist_ok=True)
        global_index = 1
        
        for vol in range(start_volume, end_volume + 1):
            url = CASE_LAW_BASE_URL.format(vol)
            print(f"\n===== Volume {vol} =====")
            print(f"Fetching: {url}")
            
            try:
                soup = self._get_soup(url)
            except Exception as e:
                print(f"  !! Cannot load volume {vol}: {e}")
                continue
            
            # Find all JSON file links
            json_files = sorted(
                href for a in soup.find_all("a", href=True)
                if (href := a["href"].lower()).endswith(".json")
            )
            
            print(f"  Found {len(json_files)} JSON files")
            
            for json_file in json_files:
                file_url = urljoin(url, json_file)
                dest_path = self.output_dir / f"{global_index}.json"
                
                print(f"    {global_index}: {file_url}")
                
                try:
                    response = requests.get(file_url, headers=self.headers)
                    response.raise_for_status()
                    with open(dest_path, "wb") as f:
                        f.write(response.content)
                    global_index += 1
                except Exception as e:
                    print(f"      !! Failed: {e}")
        
        print(f"\nDone! Downloaded {global_index - 1} files to: {self.output_dir}")
        return global_index - 1
    
    def download_from_gdrive(
        self,
        file_id: str = GDRIVE_FILE_ID,
        extract_dir: Optional[Path] = None
    ) -> Path:
        """
        Download pre-collected data from Google Drive (fallback option).
        
        Args:
            file_id: Google Drive file ID
            extract_dir: Directory to extract files to
            
        Returns:
            Path to extracted data directory
        """
        if not GDOWN_AVAILABLE:
            raise ImportError("gdown is required. Install with: pip install gdown")
        
        extract_dir = extract_dir or CASE_LAW_DIR
        zip_path = self.output_dir / "case_law.zip"
        
        # Download from Google Drive
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading from Google Drive...")
        gdown.download(url, str(zip_path), quiet=False)
        
        # Extract
        os.makedirs(extract_dir, exist_ok=True)
        print(f"Extracting to {extract_dir}...")
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        
        print(f"Done! Extracted to: {extract_dir}")
        print(f"Contents: {os.listdir(extract_dir)}")
        
        return extract_dir
    
    def create_zip_archive(self, source_dir: Path, output_path: Path) -> Path:
        """Create a ZIP archive of downloaded cases for backup."""
        import shutil
        
        archive_path = shutil.make_archive(
            str(output_path.with_suffix('')),
            'zip',
            source_dir
        )
        print(f"Created archive: {archive_path}")
        return Path(archive_path)


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download VerdictVision case law data")
    parser.add_argument("--method", choices=["api", "gdrive"], default="gdrive",
                       help="Download method: 'api' for Case.law or 'gdrive' for Google Drive")
    parser.add_argument("--output", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    collector = DataCollector(Path(args.output) if args.output else None)
    
    if args.method == "api":
        collector.download_from_case_law()
    else:
        collector.download_from_gdrive()


if __name__ == "__main__":
    main()
