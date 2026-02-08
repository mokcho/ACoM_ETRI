import os, glob, argparse, logging, math
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from baselines.sound_classification.pipelines import BaseClassifier
from baselines.beats.BEATs import BEATs, BEATsConfig
from data import *
from encodec import EncodecModel

try:
    import opuslib
    OPUS_AVAILABLE = True
except ImportError:
    OPUS_AVAILABLE = False
    logging.warning("opuslib not available, Opus codec will not work")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ------------------------------
# Helper Functions
# ------------------------------

def collate_fn(batch):
    """
    Collate function for ESCDataset which returns dictionaries
    """
    audios = []
    labels = []
    filenames = []
    
    for item in batch:
        audio = item['audio']  # Shape: (channels, samples)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=False)  # Shape: (samples,)
        else:
            audio = audio.squeeze(0)  # Shape: (samples,)
        
        audios.append(audio)
        labels.append(item['label'])
        filenames.append(item['filename'])
    
    # Get lengths before padding
    lengths = torch.tensor([len(w) for w in audios], dtype=torch.long)
    
    # Pad sequences
    wavs_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)
    
    # Add channel dimension: (B, T) -> (B, 1, T)
    wavs_padded = wavs_padded.unsqueeze(1)
    
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    return wavs_padded, labels, lengths, filenames

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _infer_grid_Fprime(T_prime, cfg, F_target):
    ps = getattr(cfg, "input_patch_size", 16)
    if isinstance(ps, (tuple, list)): ps_f = int(ps[0])
    else: ps_f = int(ps)
    F_mel = int(getattr(cfg, "input_fdim", 128))
    Fp_guess = max(1, F_mel // ps_f)

    if T_prime % Fp_guess == 0: return Fp_guess
    divisors = [d for d in range(1, min(T_prime, 64)+1) if T_prime % d == 0]
    best = min(divisors, key=lambda d: abs(d - Fp_guess))
    return best

class AudioProcessor:
    def __init__(self, cfgs, test_fold):
        self.device = DEVICE
        self.cfgs = cfgs
        self.sr = cfgs.data.sr
        
        if 'ESC-50' in cfgs.data_dir:
            self.dataset = ESCDataset(
                root_dir=cfgs.data_dir,
                annotation_dir=cfgs.annotation_dir,
                sample_rate=cfgs.data.sr,
                test_fold=test_fold,
                train=False  # Load only test fold
            )
        else:
            raise NotImplementedError
        
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.cfgs.batch_size,
            shuffle=False,
            num_workers=self.cfgs.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
            
        self.num_classes = len(self.dataset.label2id)

        self.pipeline = BaseClassifier(
            cfgs=self.cfgs, 
            label2id=self.dataset.label2id
        ).to(self.device)
        
        # Load Baseline for this fold
        path = self.cfgs.baseline.load_pretrained
        print(f"Loading pretrained model from fold {test_fold}")
        
        if path.endswith('.pt'):
            checkpoint = torch.load(path, map_location=self.device)
        else:
            checkpoint = torch.load(
                os.path.join(path, f"fold_{test_fold}", "latest.pt"), 
                map_location=self.device
            )
        
        self.pipeline.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.pipeline.eval()

        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=cfgs.data.n_fft, 
            hop_length=cfgs.data.hop, 
            power=None, 
            return_complex=True
        ).to(self.device)
        
        self.ispec = torchaudio.transforms.InverseSpectrogram(
            n_fft=cfgs.data.n_fft, 
            hop_length=cfgs.data.hop
        ).to(self.device)
        
        self.encodec = EncodecModel.encodec_model_24khz().to(self.device).eval()
        
        # Opus encoder/decoder cache
        self._opus_encoder_cache = {}
        self._opus_decoder_cache = {}

    def stft(self, wav): 
        return self.spec(wav.squeeze(1))
    
    def istft(self, S, length): 
        return self.ispec(S, length=length).unsqueeze(1)
    
    def _get_opus_encoder(self, bitrate_bps):
        """Get cached Opus encoder for a given bitrate"""
        if bitrate_bps not in self._opus_encoder_cache:
            if not OPUS_AVAILABLE:
                raise RuntimeError("opuslib is not available. Install with: pip install opuslib")
            
            encoder = opuslib.Encoder(
                fs=self.sr,
                channels=1,
                application=opuslib.APPLICATION_AUDIO
            )
            encoder.bitrate = bitrate_bps
            self._opus_encoder_cache[bitrate_bps] = encoder
        return self._opus_encoder_cache[bitrate_bps]
    
    def _get_opus_decoder(self):
        """Get cached Opus decoder"""
        if 'decoder' not in self._opus_decoder_cache:
            if not OPUS_AVAILABLE:
                raise RuntimeError("opuslib is not available. Install with: pip install opuslib")
            
            self._opus_decoder_cache['decoder'] = opuslib.Decoder(
                fs=self.sr,
                channels=1
            )
        return self._opus_decoder_cache['decoder']
    
    @torch.no_grad()
    def opus_roundtrip(self, wav16, kbps):
        """
        Encode and decode audio with Opus
        
        Args:
            wav16: torch tensor [B, 1, T] at self.sr sample rate
            kbps: target bitrate in kbps
            
        Returns:
            decoded audio as torch tensor [B, 1, T]
        """
        B, _, T_wav = wav16.shape
        if T_wav == 0:
            return torch.zeros_like(wav16)
        
        bitrate_bps = int(kbps * 1000)
        
        # Opus frame size: 20ms at sample rate
        # For 16kHz: 0.02 * 16000 = 320 samples
        frame_size = int(self.sr * 0.02)
        
        # Get cached encoder/decoder
        encoder = self._get_opus_encoder(bitrate_bps)
        decoder = self._get_opus_decoder()
        
        # Process each sample in batch
        decoded_batch = []
        
        for i in range(B):
            # Get single audio [1, T] -> [T]
            audio_np = wav16[i, 0].cpu().numpy()
            
            # Pad to multiple of frame_size
            orig_length = len(audio_np)
            pad_length = (frame_size - (orig_length % frame_size)) % frame_size
            
            if pad_length > 0:
                audio_np = np.pad(audio_np, (0, pad_length), mode='constant')
            
            # Convert to int16 for Opus
            audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
            
            # Encode frame by frame
            encoded_frames = []
            for j in range(0, len(audio_int16), frame_size):
                frame = audio_int16[j:j+frame_size]
                if len(frame) == frame_size:
                    try:
                        encoded = encoder.encode(frame.tobytes(), frame_size)
                        encoded_frames.append(encoded)
                    except Exception as e:
                        logging.error(f"Opus encoding failed: {e}")
                        # Use zeros for failed frames
                        encoded_frames.append(b'\x00' * 100)
            
            # Decode
            decoded_audio = []
            for encoded in encoded_frames:
                try:
                    decoded = decoder.decode(encoded, frame_size)
                    decoded_int16 = np.frombuffer(decoded, dtype=np.int16)
                    decoded_audio.append(decoded_int16)
                except Exception as e:
                    logging.error(f"Opus decoding failed: {e}")
                    # Use zeros for failed frames
                    decoded_audio.append(np.zeros(frame_size, dtype=np.int16))
            
            # Concatenate and convert back to float
            decoded_np = np.concatenate(decoded_audio).astype(np.float32) / 32767.0
            
            # Remove padding and restore original length
            decoded_np = decoded_np[:orig_length]
            
            # Ensure exact length match (handle any length mismatch)
            if len(decoded_np) != T_wav:
                if len(decoded_np) > T_wav:
                    decoded_np = decoded_np[:T_wav]
                else:
                    decoded_np = np.pad(decoded_np, (0, T_wav - len(decoded_np)), mode='constant')
            
            # Convert back to torch and add to batch
            decoded_torch = torch.from_numpy(decoded_np).to(wav16.device)
            decoded_batch.append(decoded_torch)
        
        # Stack batch [B, T] -> [B, 1, T]
        y16 = torch.stack(decoded_batch, dim=0).unsqueeze(1)
        
        return y16
    
    @torch.no_grad()
    def encodec_roundtrip(self, wav16, kbps):
        self.encodec.set_target_bandwidth(kbps)
        B, _, T_wav = wav16.shape
        if T_wav == 0: return torch.zeros_like(wav16)
        wav24 = AF.resample(wav16, self.sr, 24000)
        enc = self.encodec.encode(wav24)
        y24 = self.encodec.decode(enc)
        y16 = AF.resample(y24, 24000, self.sr)
        if y16.shape[-1] != T_wav:
            if y16.shape[-1] > T_wav: y16 = y16[..., :T_wav]
            else:
                pad = torch.zeros(B, 1, T_wav - y16.shape[-1], device=y16.device)
                y16 = torch.cat([y16, pad], dim=-1)
        return y16

class FeatureOnlyFilter:
    def __init__(self, processor):
        self.processor = processor
        self.device = processor.device

    def get_batch_saliency(self, wav_batch):
        with torch.no_grad():
            S_orig = self.processor.stft(wav_batch)
        
        feat = None
        def hook(m, i, o):
            nonlocal feat
            feat = o.detach()
        
        handle = self.processor.pipeline.baseline.layer_norm.register_forward_hook(hook)
        with torch.no_grad(): 
            _ = self.processor.pipeline(wav_batch, padding_mask=None)
        handle.remove()

        if feat is None: 
            return None, None

        B, F_target, T_target = S_orig.shape
        token_score = feat.abs().mean(dim=-1)
        maps_list = []
        
        for i in range(B):
            ts = token_score[i]
            F_prime = _infer_grid_Fprime(
                ts.numel(), 
                self.processor.pipeline.baseline.cfg, 
                F_target
            )
            N_prime = ts.numel() // F_prime
            grid = ts[:F_prime*N_prime].view(F_prime, N_prime)
            sal = F.interpolate(
                grid.unsqueeze(0).unsqueeze(0), 
                size=(F_target, T_target),
                mode='bilinear', 
                align_corners=False
            ).squeeze()
            
            if sal.max() > 0: 
                sal = sal / sal.max()
            maps_list.append(sal)
        
        sal_batch = torch.stack(maps_list).to(self.device)
        return S_orig, sal_batch

def process_fold_kbps(processor, saliency_filter, output_dir, step, fold, kbps, codec='encodec'):
    """
    Process all files in a specific fold at a specific kbps with specified codec
    
    Args:
        codec: 'encodec' or 'opus'
    """
    fold_kbps_results = []
    
    logging.info(f"Processing fold {fold} @ {kbps} kbps with {codec.upper()}")
    
    # Select compression function based on codec
    if codec == 'opus':
        if not OPUS_AVAILABLE:
            raise RuntimeError("Opus codec requested but opuslib is not available")
        compress_fn = processor.opus_roundtrip
    else:  # encodec
        compress_fn = processor.encodec_roundtrip
    
    for x_orig, y_true, lengths, filenames in tqdm(processor.dataloader, desc=f"Fold {fold} @ {kbps}kbps ({codec})"):
        x_orig = x_orig.to(DEVICE)
        y_true = y_true.to(DEVICE)
        lengths = lengths.to(DEVICE)
        B = x_orig.shape[0]
        
        # Get saliency maps
        S_orig, sal_map = saliency_filter.get_batch_saliency(x_orig)
        
        if sal_map is None:
            logging.warning("Saliency computation failed, skipping batch")
            continue
        
        freq_scores = sal_map.mean(dim=2)  # (B, F)
        F = freq_scores.shape[1]
        max_remove = int(F * 0.5)
        
        # For each sample in batch
        for i in range(B):
            
            filename = os.path.basename(filenames[i])
            save_path = os.path.join(output_dir, filename)
            
            # Skip if already processed
            if os.path.exists(save_path):
                logging.debug(f"Skipping {filename} - already exists")
                continue
            
            best_score = -1e9
            best_k = 0
            best_wav = None
            
            # Try different masking ratios
            for k in range(0, max_remove + 1, step):
                # Create mask
                mask = torch.ones((F, 1), device=DEVICE)
                if k > 0:
                    sorted_idx = torch.argsort(freq_scores[i], descending=False)
                    low_k_idx = sorted_idx[:k]
                    mask[low_k_idx] = 0.0
                
                # Apply mask to STFT
                S_masked = S_orig[i] * mask
                
                # Reconstruct audio
                wav = processor.istft(S_masked.unsqueeze(0), length=lengths[i])
                
                # Apply compression (EnCodec or Opus)
                wav_compressed = compress_fn(wav, kbps=kbps)
                
                # Get classifier score
                with torch.no_grad():
                    output = processor.pipeline(wav_compressed, padding_mask=None)
                    # Handle tuple output from BaseClassifier
                    if isinstance(output, tuple):
                        logits = output[0]  # First element is usually logits
                    else:
                        logits = output
                    
                    # Get score for true label (oracle)
                    score = logits[0, y_true[i]].item()
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_wav = wav_compressed.squeeze().cpu()
            
            # Save best masked audio
            torchaudio.save(save_path, best_wav.unsqueeze(0), processor.sr)
            
            # Record results
            fold_kbps_results.append({
                'filename': filename,
                'fold': fold,
                'kbps': kbps,
                'codec': codec,
                'best_masked_ratio': best_k / F,
                'masked_freq_bins': best_k,
                'total_freq_bins': F,
                'score': best_score
            })
    
    return fold_kbps_results

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", default="configs/mask_EnCodec_BEATs_ESC-50.yaml")
    ap.add_argument("--output_dir", default="/data/ESC-50-master/audio_sal_forward", help="Output directory")
    ap.add_argument("--folds", nargs="*", type=int, default=[3], help="Folds to process")
    ap.add_argument("--kbps_list", nargs="*", type=float, default=[1.5], help="kbps levels")
    ap.add_argument("--codec", type=str, default='encodec', choices=['encodec', 'opus'], 
                    help="Codec to use for compression")
    args = ap.parse_args()
    
    cfg = OmegaConf.load(args.cfgs)
    set_seed(cfg.seed)
    
    # Check Opus availability if needed
    if args.codec == 'opus' and not OPUS_AVAILABLE:
        logging.error("Opus codec requested but opuslib is not installed!")
        logging.error("Install with: pip install opuslib")
        return
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    
    all_results = []
    
    # Process each fold
    for fold in args.folds:
        logging.info(f"\n{'='*50}")
        logging.info(f"Processing Fold {fold} with {args.codec.upper()}")
        logging.info(f"{'='*50}")
        
        # Load processor for this fold
        processor = AudioProcessor(cfgs=cfg, test_fold=fold)
        saliency_filter = FeatureOnlyFilter(processor)
        
        # Process each kbps level
        for kbps in args.kbps_list:
            # Create output directory for this fold/kbps/codec combination
            fold_kbps_dir = os.path.join(args.output_dir, args.codec, f"fold_{fold}", f"kbps_{kbps}")
            os.makedirs(fold_kbps_dir, exist_ok=True)
            
            # Process this fold at this kbps
            fold_kbps_results = process_fold_kbps(
                processor=processor,
                saliency_filter=saliency_filter,
                output_dir=fold_kbps_dir,
                step=cfg.prune_step,
                fold=fold,
                kbps=kbps,
                codec=args.codec
            )
            
            all_results.extend(fold_kbps_results)
        
        # Clear memory
        del processor, saliency_filter
        torch.cuda.empty_cache()
    
    # Save combined CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, args.codec, 'masking_results.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    logging.info(f"\n{'='*50}")
    logging.info(f"ALL PROCESSING COMPLETE")
    logging.info(f"{'='*50}")
    logging.info(f"Codec: {args.codec.upper()}")
    logging.info(f"Saved {len(all_results)} masked audio files")
    logging.info(f"Saved results to {csv_path}")
    
    # Print summary statistics
    logging.info(f"\nOverall Summary:")
    logging.info(f"  Mean masking ratio: {df['best_masked_ratio'].mean():.3f}")
    logging.info(f"  Mean masked bins: {df['masked_freq_bins'].mean():.1f}")
    logging.info(f"  Mean score: {df['score'].mean():.3f}")
    
    # Per-fold, per-kbps summary
    logging.info(f"\nPer-fold, per-kbps Summary:")
    for fold in args.folds:
        for kbps in args.kbps_list:
            fold_kbps_df = df[(df['fold'] == fold) & (df['kbps'] == kbps)]
            if len(fold_kbps_df) > 0:
                logging.info(f"  Fold {fold} @ {kbps}kbps: ratio={fold_kbps_df['best_masked_ratio'].mean():.3f}, score={fold_kbps_df['score'].mean():.3f}")

if __name__ == "__main__":
    main()