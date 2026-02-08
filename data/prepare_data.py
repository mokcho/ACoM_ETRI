"""
Download Audio Datasets

Supports multiple audio datasets:
- ESC-50: Environmental sound classification
- UrbanSound8K: Urban sound classification

Usage:
    python download_audio_datasets.py --dataset esc50
    python download_audio_datasets.py --dataset urbansound8k --data_home /path/to/save
"""

import os
import urllib.request
import zipfile
import argparse


def download_esc50(save_dir='.'):
    """
    Download and extract ESC-50 dataset from GitHub
    
    Args:
        save_dir: Directory to save and extract the dataset
        
    Returns:
        extracted_path: Path to extracted dataset
    """
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = os.path.join(save_dir, "ESC-50-master.zip")
    
    print("=" * 80)
    print("ESC-50 Dataset Download")
    print("=" * 80)
    print(f"\nDataset: ESC-50")
    print(f"Source: {url}")
    print(f"Save directory: {save_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Download the zip file
    print(f"\n1. Downloading...")
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\r   Progress: {percent}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, zip_path, reporthook=progress_hook)
        print()  # New line after progress
        
        file_size = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"   ✓ Downloaded to: {zip_path}")
        print(f"   File size: {file_size:.2f} MB")
    except Exception as e:
        print(f"\n   ✗ Error downloading: {e}")
        return None
    
    # Extract the zip file
    print(f"\n2. Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
            extracted_files = zip_ref.namelist()
            print(f"   ✓ Extracted {len(extracted_files)} files")
            
            main_folder = extracted_files[0].split('/')[0]
            extracted_path = os.path.join(save_dir, main_folder)
            print(f"   Dataset location: {extracted_path}")
    except Exception as e:
        print(f"   ✗ Error extracting: {e}")
        return None
    
    # Remove the zip file
    print(f"\n3. Cleaning up...")
    try:
        os.remove(zip_path)
        print(f"   ✓ Removed: {zip_path}")
    except Exception as e:
        print(f"   ✗ Error removing zip: {e}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUCCESS!")
    print("=" * 80)
    print(f"\nDataset ready at: {extracted_path}")
    
    # Check contents
    audio_dir = os.path.join(extracted_path, 'audio')
    if os.path.exists(audio_dir):
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        print(f"Audio files: {len(audio_files)} files in {audio_dir}")
    
    meta_file = os.path.join(extracted_path, 'meta', 'esc50.csv')
    if os.path.exists(meta_file):
        print(f"Metadata: {meta_file}")
    
    return extracted_path


def download_urbansound8k(save_dir='./data'):
    """
    Download UrbanSound8K dataset using soundata
    
    Args:
        save_dir: Directory to save the dataset
        
    Returns:
        data_home: Path to dataset
    """
    print("=" * 80)
    print("UrbanSound8K Dataset Download")
    print("=" * 80)
    print(f"\nDataset: UrbanSound8K")
    print(f"Save directory: {save_dir}")
    
    # Check if soundata is installed
    try:
        import soundata
        print(f"   ✓ soundata version: {soundata.__version__}")
    except ImportError:
        print(f"\n   ✗ soundata not installed!")
        print(f"\n   Please install it first:")
        print(f"   pip install soundata")
        return None
    
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    data_home = os.path.join(save_dir, 'UrbanSound8K')
    
    print(f"\n1. Initializing dataset...")
    try:
        dataset = soundata.initialize('urbansound8k', data_home=data_home)
        print(f"   ✓ Initialized at: {data_home}")
    except Exception as e:
        print(f"   ✗ Error initializing: {e}")
        return None
    
    print(f"\n2. Downloading dataset...")
    print(f"   Note: This may take a while (~6 GB)")
    try:
        dataset.download()
        print(f"   ✓ Download complete")
    except Exception as e:
        print(f"   ✗ Error downloading: {e}")
        print(f"\n   Tip: You may need to manually download from:")
        print(f"   https://urbansounddataset.weebly.com/urbansound8k.html")
        return None
    
    print(f"\n3. Validating dataset...")
    try:
        dataset.validate()
        print(f"   ✓ All files validated successfully")
    except Exception as e:
        print(f"   ⚠️  Validation warning: {e}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUCCESS!")
    print("=" * 80)
    print(f"\nDataset ready at: {data_home}")
    
    # Test loading a sample
    try:
        example_clip = dataset.choice_clip()
        print(f"\nExample clip:")
        print(f"  ID: {example_clip.clip_id}")
        print(f"  Class: {example_clip.class_name}")
        print(f"  Audio path: {example_clip.audio_path}")
    except Exception as e:
        print(f"\n⚠️  Could not load example: {e}")
    
    return data_home


def download_dataset(dataset_name, save_dir=None):
    """
    Download specified audio dataset
    
    Args:
        dataset_name: Name of dataset ('esc50' or 'urbansound8k')
        save_dir: Directory to save dataset (default varies by dataset)
        
    Returns:
        dataset_path: Path to downloaded dataset
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name in ['esc50', 'esc-50']:
        if save_dir is None:
            save_dir = '.'
        return download_esc50(save_dir)
    
    elif dataset_name in ['urbansound8k', 'urbansound', 'us8k']:
        if save_dir is None:
            save_dir = './data'
        return download_urbansound8k(save_dir)
    
    else:
        print(f"✗ Unknown dataset: {dataset_name}")
        print(f"\nSupported datasets:")
        print(f"  - esc50 (or esc-50)")
        print(f"  - urbansound8k (or urbansound, us8k)")
        return None


def main():
    parser = argparse.ArgumentParser(description='Download audio datasets')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['esc50', 'esc-50', 'urbansound8k', 'urbansound', 'us8k'],
                       help='Dataset to download')
    parser.add_argument('--data_home', type=str, default=None,
                       help='Directory to save dataset (default: varies by dataset)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("AUDIO DATASET DOWNLOADER")
    print("=" * 80)
    
    dataset_path = download_dataset(args.dataset, args.data_home)
    
    if dataset_path:
        print(f"\n{'=' * 80}")
        print("USAGE EXAMPLE")
        print("=" * 80)
        
        if 'esc' in args.dataset.lower():
            print("\nLoad ESC-50 audio:")
            print("  import librosa")
            print(f"  audio, sr = librosa.load('{dataset_path}/audio/1-100032-A-0.wav', sr=16000)")
            print("\nLoad metadata:")
            print("  import pandas as pd")
            print(f"  df = pd.read_csv('{dataset_path}/meta/esc50.csv')")
        
        elif 'urbansound' in args.dataset.lower():
            print("\nLoad UrbanSound8K:")
            print("  import soundata")
            print(f"  dataset = soundata.initialize('urbansound8k', data_home='{dataset_path}')")
            print("  clip = dataset.choice_clip()")
            print("  audio, sr = clip.audio")
    else:
        print("\n✗ Download failed!")


if __name__ == "__main__":
    # If no arguments provided, show interactive menu
    import sys
    if len(sys.argv) == 1:
        print("=" * 80)
        print("AUDIO DATASET DOWNLOADER")
        print("=" * 80)
        print("\nAvailable datasets:")
        print("  1. ESC-50 (Environmental Sound Classification)")
        print("  2. UrbanSound8K (Urban Sound Classification)")
        
        choice = input("\nSelect dataset (1 or 2): ").strip()
        
        if choice == '1':
            save_dir = input("Save directory (default: current dir): ").strip() or '.'
            download_dataset('esc50', save_dir)
        elif choice == '2':
            save_dir = input("Save directory (default: ./data): ").strip() or './data'
            download_dataset('urbansound8k', save_dir)
        else:
            print("Invalid choice!")
    else:
        main()