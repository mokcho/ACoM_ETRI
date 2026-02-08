"""
Download BEATs Model Checkpoint

Downloads BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt1.pt from OneDrive
"""

import os
import urllib.request
import sys


def download_beats_checkpoint(save_dir='./BEATs'):
    """
    Download BEATs checkpoint from OneDrive
    
    Args:
        save_dir: Directory to save the checkpoint
        
    Returns:
        checkpoint_path: Path to downloaded checkpoint
    """
    # OneDrive direct download link
    # Original link: https://1drv.ms/u/s!AqeByhGUtINrgcpoZecQbiXeaUjN8A?e=DasbeC
    # Direct download link (converted):
    url = "https://onedrive.live.com/download?cid=9DA444B7D5BBD90E&resid=9DA444B7D5BBD90E%21677&authkey=AOPLaGXnEG4l3mk"
    
    checkpoint_name = "BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt1.pt"
    
    print("=" * 80)
    print("BEATs Model Checkpoint Download")
    print("=" * 80)
    print(f"\nModel: BEATs iter3+ (AS20K finetuned on AS2M)")
    print(f"Source: OneDrive")
    print(f"Save directory: {save_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    
    # Check if already downloaded
    if os.path.exists(checkpoint_path):
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"\n✓ Checkpoint already exists: {checkpoint_path}")
        print(f"  File size: {file_size:.2f} MB")
        
        response = input("\nRe-download? (y/n): ").strip().lower()
        if response != 'y':
            print("Using existing checkpoint.")
            return checkpoint_path
    
    # Download the checkpoint
    print(f"\nDownloading checkpoint...")
    print(f"This may take a while (~300 MB)")
    
    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                downloaded = count * block_size / (1024 * 1024)
                total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent}% ({downloaded:.1f}/{total:.1f} MB)", end='', flush=True)
            else:
                downloaded = count * block_size / (1024 * 1024)
                print(f"\r  Downloaded: {downloaded:.1f} MB", end='', flush=True)
        
        urllib.request.urlretrieve(url, checkpoint_path, reporthook=progress_hook)
        print()  # New line after progress
        
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"\n✓ Downloaded successfully!")
        print(f"  Saved to: {checkpoint_path}")
        print(f"  File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"\n✗ Error downloading: {e}")
        print(f"\nIf the automatic download fails, please:")
        print(f"1. Manually download from: https://1drv.ms/u/s!AqeByhGUtINrgcpoZecQbiXeaUjN8A?e=DasbeC")
        print(f"2. Save it as: {checkpoint_path}")
        return None
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUCCESS!")
    print("=" * 80)
    print(f"\nCheckpoint ready at: {checkpoint_path}")
    
    return checkpoint_path


def download_all_beats_checkpoints(save_dir='./BEATs'):
    """
    Download all available BEATs checkpoints
    
    Available models:
    - BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt1.pt (recommended)
    - BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
    - BEATs_iter3_plus_AS20K.pt
    """
    print("=" * 80)
    print("BEATs Model Checkpoints")
    print("=" * 80)
    print("\nAvailable checkpoints:")
    print("  1. BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt1.pt (recommended)")
    print("     - Pretrained on AudioSet-20K, finetuned on AudioSet-2M")
    print("     - Best for general audio tasks")
    print("\n  Note: Other checkpoints require different download links")
    
    choice = input("\nDownload checkpoint 1? (y/n): ").strip().lower()
    
    if choice == 'y':
        return download_beats_checkpoint(save_dir)
    else:
        print("Download cancelled.")
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download BEATs model checkpoint')
    parser.add_argument('--save_dir', type=str, default='/data/BEATs',
                       help='Directory to save checkpoint (default: /data/BEATs)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if file exists')
    
    args = parser.parse_args()
    
    checkpoint_path = download_beats_checkpoint(args.save_dir)
    
    if checkpoint_path:
        print(f"\n{'=' * 80}")
        print("USAGE EXAMPLE")
        print("=" * 80)
        print("\nLoad the checkpoint in Python:")
        print("  import torch")
        print(f"  checkpoint = torch.load('{checkpoint_path}')")
        print("  print(checkpoint.keys())")
        print("\nOr with BEATs model:")
        print("  from BEATs import BEATs, BEATsConfig")
        print(f"  checkpoint = torch.load('{checkpoint_path}')")
        print("  cfg = BEATsConfig(checkpoint['cfg'])")
        print("  model = BEATs(cfg)")
        print("  model.load_state_dict(checkpoint['model'])")


if __name__ == "__main__":
    import sys
    
    # If no arguments, show interactive prompt
    if len(sys.argv) == 1:
        checkpoint_path = download_beats_checkpoint('./BEATs')
        
        if checkpoint_path:
            print(f"\n{'=' * 80}")
            print("USAGE EXAMPLE")
            print("=" * 80)
            print("\nLoad the checkpoint:")
            print("  import torch")
            print(f"  checkpoint = torch.load('{checkpoint_path}')")
    else:
        main()