# import os, glob, argparse, logging, math
# from collections import defaultdict
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from pathlib import Path

# from omegaconf import OmegaConf

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchaudio
# import torchaudio.functional as AF

# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence

# # ESC-50 imports
# from baselines.sound_classification.pipelines import BaseClassifier
# from data import ESCDataset, collate_fn_multi_class

# # AudioSet imports
# from baselines.sound_event_detection.pipelines import BaseEventDetector, EnCodecEventDetector, FilterEventDetector
# from data.dataset import StronglyAnnotatedSet, WeakSet
# from data.ManyHotEncoder import ManyHotEncoder

# from encodec import EncodecModel

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# # ------------------------------
# # Helper Functions
# # ------------------------------
# def set_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)

# def _infer_grid_Fprime(T_prime, cfg, F_target):
#     ps = getattr(cfg, "input_patch_size", 16)
#     if isinstance(ps, (tuple, list)): ps_f = int(ps[0])
#     else: ps_f = int(ps)
#     F_mel = int(getattr(cfg, "input_fdim", 128))
#     Fp_guess = max(1, F_mel // ps_f)

#     if T_prime % Fp_guess == 0: return Fp_guess
#     divisors = [d for d in range(1, min(T_prime, 64)+1) if T_prime % d == 0]
#     best = min(divisors, key=lambda d: abs(d - Fp_guess))
#     return best

# def to_clip_labels(labels, num_classes):
#     """Convert frame-level labels to clip-level for AudioSet"""
#     assert labels.dim() == 3, f"expected 3D labels, got {labels.shape}"
#     if labels.shape[1] == num_classes:  # [B, C, T]
#         clip = labels.amax(dim=2)       # [B, C]
#     else:                               # [B, T, C]
#         clip = labels.transpose(1, 2).amax(dim=2)  # -> [B, C]
#     return clip

# # ------------------------------
# # Unified AudioProcessor
# # ------------------------------
# class AudioProcessor:
#     def __init__(self, cfgs, test_fold=None):
#         self.device = DEVICE
#         self.cfgs = cfgs
#         self.dataset_type = self._detect_dataset_type()
        
#         # Initialize dataset and model based on type
#         if self.dataset_type == 'esc50':
#             self._init_esc50(test_fold)
#         elif self.dataset_type == 'audioset':
#             self._init_audioset()
#         else:
#             raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        
#         # Common components - handle different config structures
#         n_fft = getattr(self.cfgs.data, 'n_fft', 1024)
#         hop = getattr(self.cfgs.data, 'hop', 256)
        
#         self.spec = torchaudio.transforms.Spectrogram(
#             n_fft=n_fft, 
#             hop_length=hop, 
#             power=None, 
#             return_complex=True
#         ).to(self.device)
        
#         self.ispec = torchaudio.transforms.InverseSpectrogram(
#             n_fft=n_fft, 
#             hop_length=hop
#         ).to(self.device)
        
#         self.encodec = EncodecModel.encodec_model_24khz().to(self.device).eval()
    
#     def _detect_dataset_type(self):
#         """Detect dataset type from config"""
#         if 'ESC-50' in self.cfgs.data_dir:
#             return 'esc50'
#         elif 'dcase' in self.cfgs.data_dir or 'audioset' in self.cfgs.annotation_dir.lower() or hasattr(self.cfgs, 'mode'):
#             return 'audioset'
#         else:
#             raise ValueError("Cannot detect dataset type from config")
    
#     def _init_esc50(self, test_fold):
#         """Initialize ESC-50 dataset and classifier"""
#         self.sr = self.cfgs.data.sr
#         self.num_classes = len(self.cfgs.data.classes) if hasattr(self.cfgs.data, 'classes') else 50
        
#         self.dataset = ESCDataset(
#             root_dir=self.cfgs.data_dir,
#             annotation_dir=self.cfgs.annotation_dir,
#             sample_rate=self.sr,
#             test_fold=test_fold,
#             train=False
#         )
        
#         self.dataloader = DataLoader(
#             dataset=self.dataset,
#             batch_size=self.cfgs.batch_size,
#             shuffle=False,
#             num_workers=self.cfgs.num_workers,
#             pin_memory=True,
#             collate_fn=collate_fn_multi_class
#         )
        
#         self.pipeline = BaseClassifier(
#             cfgs=self.cfgs, 
#             label2id=self.dataset.label2id
#         ).to(self.device)
        
#         # Load checkpoint
#         path = self.cfgs.baseline.load_pretrained
#         logging.info(f"Loading ESC-50 model from fold {test_fold}")
        
#         if path.endswith('.pt'):
#             checkpoint = torch.load(path, map_location=self.device)
#         else:
#             checkpoint = torch.load(
#                 os.path.join(path, f"fold_{test_fold}", "latest.pt"), 
#                 map_location=self.device
#             )
        
#         self.pipeline.load_state_dict(checkpoint['model_state_dict'], strict=False)
#         self.pipeline.eval()
        
#         # Hook point for ESC-50
#         self.hook_module = self.pipeline.baseline.layer_norm
#         self.beats_cfg = self.pipeline.baseline.cfg
    
#     def _init_audioset(self):
#         """Initialize AudioSet dataset and event detector"""
#         self.sr = self.cfgs.data.sr
        
#         # Load metadata - handle eval set
#         eval_annotation = self.cfgs.annotation_dir.replace("train", "eval")
#         val_tsv = pd.read_csv(eval_annotation, sep="\t")
        
#         # Initialize encoder
#         self.encoder = ManyHotEncoder(
#             labels=self.cfgs.data.classes,
#             n_frames=self.cfgs.data.n_frames
#         )
        
#         self.num_classes = len(self.cfgs.data.classes)
        
#         # Create dataset - handle eval folder structure
#         eval_data_dir = self.cfgs.data_dir.replace("train", "eval")
        
#         self.dataset = StronglyAnnotatedSet(
#             audio_folder=Path(eval_data_dir) / "strong_label_real",
#             tsv_entries=val_tsv,
#             encoder=self.encoder,
#             pad_to=self.cfgs.data.audio_max_len,
#             fs=self.cfgs.data.fs,
#             return_filename=True,
#             test=True
#         )
        
#         self.dataloader = DataLoader(
#             self.dataset,
#             batch_size=self.cfgs.batch_size,
#             shuffle=False,
#             num_workers=self.cfgs.num_workers,
#             pin_memory=True
#         )
        
#         # Initialize model based on config
#         if hasattr(self.cfgs, 'filters') and self.cfgs.filters:
#             self.pipeline = FilterEventDetector(
#                 self.cfgs,
#                 num_classes=self.num_classes,
#                 mode=self.cfgs.mode
#             ).to(self.device)
#         elif hasattr(self.cfgs, 'encodec') and self.cfgs.encodec:
#             self.pipeline = EnCodecEventDetector(
#                 self.cfgs,
#                 num_classes=self.num_classes,
#                 mode=self.cfgs.mode
#             ).to(self.device)
#         else:
#             self.pipeline = BaseEventDetector(
#                 self.cfgs,
#                 num_classes=self.num_classes,
#                 mode=self.cfgs.mode
#             ).to(self.device)
        
#         # Load checkpoint
#         path = self.cfgs.baseline.load_pretrained
#         logging.info(f"Loading AudioSet model from {path}")
        
#         checkpoint = torch.load(path, map_location=self.device)
#         self.pipeline.load_state_dict(checkpoint['strong_model_state_dict'], strict=False)
#         self.pipeline.eval()
        
#         # Hook point for AudioSet
#         self.hook_module = self.pipeline.baseline.layer_norm
#         self.beats_cfg = self.pipeline.baseline.cfg

#     def stft(self, wav): 
#         return self.spec(wav.squeeze(1))
    
#     def istft(self, S, length): 
#         return self.ispec(S, length=length).unsqueeze(1)
    
#     @torch.no_grad()
#     def encodec_roundtrip(self, wav16, kbps):
#         self.encodec.set_target_bandwidth(kbps)
#         B, _, T_wav = wav16.shape
#         if T_wav == 0: return torch.zeros_like(wav16)
#         wav24 = AF.resample(wav16, self.sr, 24000)
#         enc = self.encodec.encode(wav24)
#         y24 = self.encodec.decode(enc)
#         y16 = AF.resample(y24, 24000, self.sr)
#         if y16.shape[-1] != T_wav:
#             if y16.shape[-1] > T_wav: y16 = y16[..., :T_wav]
#             else:
#                 pad = torch.zeros(B, 1, T_wav - y16.shape[-1], device=y16.device)
#                 y16 = torch.cat([y16, pad], dim=-1)
#         return y16

# # ------------------------------
# # Saliency Filter
# # ------------------------------
# class FeatureOnlyFilter:
#     def __init__(self, processor):
#         self.processor = processor
#         self.device = processor.device

#     def get_batch_saliency(self, wav_batch):
#         with torch.no_grad():
#             S_orig = self.processor.stft(wav_batch)
        
#         feat = None
#         def hook(m, i, o):
#             nonlocal feat
#             feat = o.detach()
        
#         handle = self.processor.hook_module.register_forward_hook(hook)
#         with torch.no_grad(): 
#             _ = self.processor.pipeline(wav_batch, padding_mask=None)
#         handle.remove()

#         if feat is None: 
#             return None, None

#         B, F_target, T_target = S_orig.shape
#         token_score = feat.abs().mean(dim=-1)
#         maps_list = []
        
#         for i in range(B):
#             ts = token_score[i]
#             F_prime = _infer_grid_Fprime(
#                 ts.numel(), 
#                 self.processor.beats_cfg, 
#                 F_target
#             )
#             N_prime = ts.numel() // F_prime
#             grid = ts[:F_prime*N_prime].view(F_prime, N_prime)
#             sal = F.interpolate(
#                 grid.unsqueeze(0).unsqueeze(0), 
#                 size=(F_target, T_target),
#                 mode='bilinear', 
#                 align_corners=False
#             ).squeeze()
            
#             if sal.max() > 0: 
#                 sal = sal / sal.max()
#             maps_list.append(sal)
        
#         sal_batch = torch.stack(maps_list).to(self.device)
#         return S_orig, sal_batch

# # ------------------------------
# # Processing Functions
# # ------------------------------
# def process_fold_kbps(processor, saliency_filter, output_dir, step, fold_or_split, kbps):
#     results = []
    
#     logging.info(f"Processing {fold_or_split} @ {kbps} kbps")
    
#     pbar = tqdm(processor.dataloader, desc=f"{fold_or_split} @ {kbps}kbps")
    
#     # ADD DEBUG CHECK HERE:
#     first_batch = True
    
#     for batch_data in pbar:
        
#         # Debug: Check batch structure on first iteration
#         if first_batch:
#             logging.info(f"Batch length: {len(batch_data)}")
#             if len(batch_data) > 3:
#                 logging.info(f"Filenames found: {batch_data[3][0] if batch_data[3] else 'None'}")
#             else:
#                 logging.warning("⚠️ Filenames NOT in batch! Files will be saved as sample_N.wav")
#             first_batch = False
        
#         x_orig = batch_data[0].to(DEVICE)
#         y_true = batch_data[1].to(DEVICE)
        
#         # For AudioSet, convert frame-level to clip-level
#         if processor.dataset_type == 'audioset':
#             y_true_clip = to_clip_labels(y_true, processor.num_classes)
#             filenames = batch_data[3] if len(batch_data) > 3 else None
#             lengths = torch.full((x_orig.shape[0],), x_orig.shape[-1], device=DEVICE, dtype=torch.long)
#         else:  # ESC-50
#             y_true_clip = y_true
#             lengths = batch_data[2].to(DEVICE)
#             filenames = batch_data[3] if len(batch_data) > 3 else None
        
#         B = x_orig.shape[0]
        
#         # Get saliency maps
#         S_orig, sal_map = saliency_filter.get_batch_saliency(x_orig)
        
#         if sal_map is None:
#             logging.warning("Saliency computation failed, skipping batch")
#             continue
        
#         freq_scores = sal_map.mean(dim=2)  # (B, F)
#         F = freq_scores.shape[1]
#         max_remove = int(F * 0.5)
        
#         # For each sample in batch
#         for i in range(B):
#             best_score = -1e9
#             best_k = 0
#             best_wav = None
            
#             # Try different masking ratios
#             for k in range(0, max_remove + 1, step):
#                 # Create mask
#                 mask = torch.ones((F, 1), device=DEVICE)
#                 if k > 0:
#                     sorted_idx = torch.argsort(freq_scores[i], descending=False)
#                     low_k_idx = sorted_idx[:k]
#                     mask[low_k_idx] = 0.0
                
#                 # Apply mask to STFT
#                 S_masked = S_orig[i] * mask
                
#                 # Reconstruct audio
#                 wav = processor.istft(S_masked.unsqueeze(0), length=lengths[i])
                
#                 # Apply EnCodec compression
#                 wav_compressed = processor.encodec_roundtrip(wav, kbps=kbps)
                
#                 # Get classifier score
#                 with torch.no_grad():
#                     output = processor.pipeline(wav_compressed, padding_mask=None)
                    
#                     # Handle tuple output
#                     if isinstance(output, tuple):
#                         logits = output[0]
#                     else:
#                         logits = output
                    
#                     # Calculate score based on dataset type
#                     if processor.dataset_type == 'esc50':
#                         # Single-label: score for true class
#                         score = logits[0, y_true_clip[i]].item()
#                     else:  # audioset
#                         true_classes = y_true_clip[i].nonzero(as_tuple=True)[0]
#                         false_classes = (y_true_clip[i] == 0).nonzero(as_tuple=True)[0]
                        
#                         if len(true_classes) > 0 and len(false_classes) > 0:
#                             true_score = logits[0, true_classes].mean()
#                             false_score = logits[0, false_classes].mean()
                            
#                             # Maximize margin between true and false classes
#                             score = (true_score - false_score).item()
#                         elif len(true_classes) > 0:
#                             score = logits[0, true_classes].mean().item()
#                         else:
#                             score = logits[0].max().item()
#                     # else:  # audioset - weighted scoring
#                     #     probs = torch.sigmoid(logits[0])
                        
#                     #     # Positive weight: how well we predict true classes
#                     #     true_classes = y_true_clip[i].nonzero(as_tuple=True)[0]
#                     #     false_classes = (y_true_clip[i] == 0).nonzero(as_tuple=True)[0]
                        
#                     #     if len(true_classes) > 0:
#                     #         # Want probabilities close to 1 for true classes
#                     #         true_score = probs[true_classes].mean()
                            
#                     #         # Want probabilities close to 0 for false classes
#                     #         if len(false_classes) > 0:
#                     #             # Use 1 - prob for false classes (so lower prob = higher score)
#                     #             false_score = (1 - probs[false_classes]).mean()
#                     #         else:
#                     #             false_score = 1.0
                            
#                     #         # Geometric mean of both (balanced precision/recall)
#                     #         score = torch.sqrt(true_score * false_score).item()
#                     #     else:
#                     #         score = probs.max().item()
                
#                 # Update best
#                 if score > best_score:
#                     best_score = score
#                     best_k = k
#                     best_wav = wav_compressed.squeeze().cpu()
                    
#                     # Update progress bar as we find better scores
#                     pbar.set_postfix({
#                         'best': f'{best_score:.3f}',
#                         'k': k,
#                         'ratio': f'{k/F:.2f}'
#                     })
            
#             # Save best masked audio
#             if filenames:
#                 filename = os.path.basename(filenames[i])
#             else:
#                 filename = f"sample_{i}.wav"
                
#             save_path = os.path.join(output_dir, filename)
#             torchaudio.save(save_path, best_wav.unsqueeze(0), processor.sr)
            
#             # Record results
#             results.append({
#                 'filename': filename,
#                 'fold_or_split': fold_or_split,
#                 'kbps': kbps,
#                 'best_masked_ratio': best_k / F,
#                 'masked_freq_bins': best_k,
#                 'total_freq_bins': F,
#                 'score': best_score,
#                 'dataset_type': processor.dataset_type
#             })
    
#     return results

# # ------------------------------
# # Main
# # ------------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--cfgs", default="configs/mask_EnCodec_BEATs_AudioSet.yaml", help="Path to config file")
#     ap.add_argument("--output_dir", default="/data/dcase/dataset/encodec_masked_audio_output/strong_label_real_f1", help="Output directory")
#     ap.add_argument("--folds", nargs="*", type=int, default=None, help="Folds to process (ESC-50 only)")
#     ap.add_argument("--kbps_list", nargs="*", type=float, default=[1.5, 3.0, 6.0, 12.0, 24.0], help="kbps levels")
#     ap.add_argument("--step", type=int, default=None, help="Override prune_step from config")
#     args = ap.parse_args()
    
#     cfg = OmegaConf.load(args.cfgs)
#     set_seed(cfg.seed)
    
#     # Override step if provided
#     if args.step is not None:
#         cfg.prune_step = args.step
    
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    
#     all_results = []
    
#     # Detect dataset type and process accordingly
#     if 'ESC-50' in cfg.data_dir:
#         # ESC-50: process by folds
#         folds = args.folds if args.folds else [1, 2, 3, 4, 5]
        
#         for fold in folds:
#             logging.info(f"\n{'='*50}")
#             logging.info(f"Processing Fold {fold}")
#             logging.info(f"{'='*50}")
            
#             processor = AudioProcessor(cfgs=cfg, test_fold=fold)
#             saliency_filter = FeatureOnlyFilter(processor)
            
#             for kbps in args.kbps_list:
#                 fold_kbps_dir = os.path.join(args.output_dir, f"fold_{fold}", f"{kbps}kbps")
#                 os.makedirs(fold_kbps_dir, exist_ok=True)
                
#                 fold_results = process_fold_kbps(
#                     processor=processor,
#                     saliency_filter=saliency_filter,
#                     output_dir=fold_kbps_dir,
#                     step=cfg.prune_step,
#                     fold_or_split=f"fold_{fold}",
#                     kbps=kbps
#                 )
                
#                 all_results.extend(fold_results)
            
#             del processor, saliency_filter
#             torch.cuda.empty_cache()
    
#     else:  # AudioSet
#         logging.info(f"\n{'='*50}")
#         logging.info(f"Processing AudioSet Eval Set")
#         logging.info(f"{'='*50}")
        
#         processor = AudioProcessor(cfgs=cfg)
#         saliency_filter = FeatureOnlyFilter(processor)
        
#         for kbps in args.kbps_list:
#             eval_kbps_dir = os.path.join(args.output_dir, f"{kbps}kbps")
#             os.makedirs(eval_kbps_dir, exist_ok=True)
            
#             eval_results = process_fold_kbps(
#                 processor=processor,
#                 saliency_filter=saliency_filter,
#                 output_dir=eval_kbps_dir,
#                 step=cfg.prune_step,
#                 fold_or_split="eval",
#                 kbps=kbps
#             )
            
#             all_results.extend(eval_results)
        
#         del processor, saliency_filter
#         torch.cuda.empty_cache()
    
#     # Save combined CSV
#     df = pd.DataFrame(all_results)
#     csv_path = os.path.join(args.output_dir, 'masking_results.csv')
#     df.to_csv(csv_path, index=False)
    
#     logging.info(f"\n{'='*50}")
#     logging.info(f"ALL PROCESSING COMPLETE")
#     logging.info(f"{'='*50}")
#     logging.info(f"Saved {len(all_results)} masked audio files")
#     logging.info(f"Saved results to {csv_path}")
    
#     # Print summary
#     logging.info(f"\nOverall Summary:")
#     logging.info(f"  Dataset type: {df['dataset_type'].iloc[0]}")
#     logging.info(f"  Mean masking ratio: {df['best_masked_ratio'].mean():.3f}")
#     logging.info(f"  Mean masked bins: {df['masked_freq_bins'].mean():.1f}")
#     logging.info(f"  Mean score: {df['score'].mean():.3f}")
    
#     # Per-split summary
#     for split in df['fold_or_split'].unique():
#         split_df = df[df['fold_or_split'] == split]
#         for kbps in args.kbps_list:
#             kbps_df = split_df[split_df['kbps'] == kbps]
#             if len(kbps_df) > 0:
#                 logging.info(f"  {split} @ {kbps}kbps: ratio={kbps_df['best_masked_ratio'].mean():.3f}, score={kbps_df['score'].mean():.3f}")

# if __name__ == "__main__":
#     main()

import os, glob, argparse, logging, math
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ESC-50 imports
from baselines.sound_classification.pipelines import BaseClassifier
from data import ESCDataset, collate_fn_multi_class

# AudioSet imports
from baselines.sound_event_detection.pipelines import BaseEventDetector, EnCodecEventDetector, FilterEventDetector
from data.dataset import StronglyAnnotatedSet, WeakSet
from data.ManyHotEncoder import ManyHotEncoder

from encodec import EncodecModel
import opuslib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ------------------------------
# Helper Functions
# ------------------------------
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

def to_clip_labels(labels, num_classes):
    """Convert frame-level labels to clip-level for AudioSet"""
    assert labels.dim() == 3, f"expected 3D labels, got {labels.shape}"
    if labels.shape[1] == num_classes:  # [B, C, T]
        clip = labels.amax(dim=2)       # [B, C]
    else:                               # [B, T, C]
        clip = labels.transpose(1, 2).amax(dim=2)  # -> [B, C]
    return clip

# ------------------------------
# Unified AudioProcessor
# ------------------------------
class AudioProcessor:
    def __init__(self, cfgs, test_fold=None):
        self.device = DEVICE
        self.cfgs = cfgs
        self.dataset_type = self._detect_dataset_type()
        
        # Initialize dataset and model based on type
        if self.dataset_type == 'esc50':
            self._init_esc50(test_fold)
        elif self.dataset_type == 'audioset':
            self._init_audioset()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        
        # Common components - handle different config structures
        n_fft = getattr(self.cfgs.data, 'n_fft', 1024)
        hop = getattr(self.cfgs.data, 'hop', 256)
        
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, 
            hop_length=hop, 
            power=None, 
            return_complex=True
        ).to(self.device)
        
        self.ispec = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft, 
            hop_length=hop
        ).to(self.device)
        
        self.encodec = EncodecModel.encodec_model_24khz().to(self.device).eval()
        
        # Opus encoder/decoder cache
        self._opus_encoder_cache = {}
        self._opus_decoder_cache = {}
    
    def _detect_dataset_type(self):
        """Detect dataset type from config"""
        if 'ESC-50' in self.cfgs.data_dir:
            return 'esc50'
        elif 'dcase' in self.cfgs.data_dir or 'audioset' in self.cfgs.annotation_dir.lower() or hasattr(self.cfgs, 'mode'):
            return 'audioset'
        else:
            raise ValueError("Cannot detect dataset type from config")
    
    def _init_esc50(self, test_fold):
        """Initialize ESC-50 dataset and classifier"""
        self.sr = self.cfgs.data.sr
        self.num_classes = len(self.cfgs.data.classes) if hasattr(self.cfgs.data, 'classes') else 50
        
        self.dataset = ESCDataset(
            root_dir=self.cfgs.data_dir,
            annotation_dir=self.cfgs.annotation_dir,
            sample_rate=self.sr,
            test_fold=test_fold,
            train=False
        )
        
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.cfgs.batch_size,
            shuffle=False,
            num_workers=self.cfgs.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_multi_class
        )
        
        self.pipeline = BaseClassifier(
            cfgs=self.cfgs, 
            label2id=self.dataset.label2id
        ).to(self.device)
        
        # Load checkpoint
        path = self.cfgs.baseline.load_pretrained
        logging.info(f"Loading ESC-50 model from fold {test_fold}")
        
        if path.endswith('.pt'):
            checkpoint = torch.load(path, map_location=self.device)
        else:
            checkpoint = torch.load(
                os.path.join(path, f"fold_{test_fold}", "latest.pt"), 
                map_location=self.device
            )
        
        self.pipeline.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.pipeline.eval()
        
        # Hook point for ESC-50
        self.hook_module = self.pipeline.baseline.layer_norm
        self.beats_cfg = self.pipeline.baseline.cfg
    
    def _init_audioset(self):
        """Initialize AudioSet dataset and event detector"""
        self.sr = self.cfgs.data.sr
        
        # Load metadata - handle eval set
        eval_annotation = self.cfgs.annotation_dir.replace("train", "eval")
        val_tsv = pd.read_csv(eval_annotation, sep="\t")
        
        # Initialize encoder
        self.encoder = ManyHotEncoder(
            labels=self.cfgs.data.classes,
            n_frames=self.cfgs.data.n_frames
        )
        
        self.num_classes = len(self.cfgs.data.classes)
        
        # Create dataset - handle eval folder structure
        eval_data_dir = self.cfgs.data_dir.replace("train", "eval")
        
        self.dataset = StronglyAnnotatedSet(
            audio_folder=Path(eval_data_dir) / "strong_label_real",
            tsv_entries=val_tsv,
            encoder=self.encoder,
            pad_to=self.cfgs.data.audio_max_len,
            fs=self.cfgs.data.fs,
            return_filename=True,
            test=True
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfgs.batch_size,
            shuffle=False,
            num_workers=self.cfgs.num_workers,
            pin_memory=True
        )
        
        # Initialize model based on config
        if hasattr(self.cfgs, 'filters') and self.cfgs.filters:
            self.pipeline = FilterEventDetector(
                self.cfgs,
                num_classes=self.num_classes,
                mode=self.cfgs.mode
            ).to(self.device)
        elif hasattr(self.cfgs, 'encodec') and self.cfgs.encodec:
            self.pipeline = EnCodecEventDetector(
                self.cfgs,
                num_classes=self.num_classes,
                mode=self.cfgs.mode
            ).to(self.device)
        else:
            self.pipeline = BaseEventDetector(
                self.cfgs,
                num_classes=self.num_classes,
                mode=self.cfgs.mode
            ).to(self.device)
        
        # Load checkpoint
        path = self.cfgs.baseline.load_pretrained
        logging.info(f"Loading AudioSet model from {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.pipeline.load_state_dict(checkpoint['strong_model_state_dict'], strict=False)
        self.pipeline.eval()
        
        # Hook point for AudioSet
        self.hook_module = self.pipeline.baseline.layer_norm
        self.beats_cfg = self.pipeline.baseline.cfg

    def stft(self, wav): 
        return self.spec(wav.squeeze(1))
    
    def istft(self, S, length): 
        return self.ispec(S, length=length).unsqueeze(1)
    
    def _get_opus_encoder(self, bitrate_bps):
        """Get cached Opus encoder for a given bitrate"""
        if bitrate_bps not in self._opus_encoder_cache:
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
        frame_size = int(self.sr * 0.02)  # 20ms frames at 16kHz = 320 samples
        
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
                    encoded = encoder.encode(frame.tobytes(), frame_size)
                    encoded_frames.append(encoded)
            
            # Decode
            decoded_audio = []
            for encoded in encoded_frames:
                decoded = decoder.decode(encoded, frame_size)
                decoded_int16 = np.frombuffer(decoded, dtype=np.int16)
                decoded_audio.append(decoded_int16)
            
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

# ------------------------------
# Saliency Filter
# ------------------------------
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
        
        handle = self.processor.hook_module.register_forward_hook(hook)
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
                self.processor.beats_cfg, 
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

# ------------------------------
# Processing Functions
# ------------------------------
def process_fold_kbps(processor, saliency_filter, output_dir, step, fold_or_split, kbps, codec='encodec'):
    """
    Process fold with specified codec
    
    Args:
        codec: 'encodec' or 'opus'
    """
    results = []
    
    logging.info(f"Processing {fold_or_split} @ {kbps} kbps with {codec.upper()}")
    
    pbar = tqdm(processor.dataloader, desc=f"{fold_or_split} @ {kbps}kbps ({codec})")
    
    # Select compression function
    if codec == 'opus':
        compress_fn = processor.opus_roundtrip
    else:  # encodec
        compress_fn = processor.encodec_roundtrip
    
    first_batch = True
    
    for batch_data in pbar:
        
        # Debug: Check batch structure on first iteration
        if first_batch:
            logging.info(f"Batch length: {len(batch_data)}")
            if len(batch_data) > 3:
                logging.info(f"Filenames found: {batch_data[3][0] if batch_data[3] else 'None'}")
            else:
                logging.warning("⚠️ Filenames NOT in batch! Files will be saved as sample_N.wav")
            first_batch = False
        
        x_orig = batch_data[0].to(DEVICE)
        y_true = batch_data[1].to(DEVICE)
        
        # For AudioSet, convert frame-level to clip-level
        if processor.dataset_type == 'audioset':
            y_true_clip = to_clip_labels(y_true, processor.num_classes)
            filenames = batch_data[3] if len(batch_data) > 3 else None
            lengths = torch.full((x_orig.shape[0],), x_orig.shape[-1], device=DEVICE, dtype=torch.long)
        else:  # ESC-50
            y_true_clip = y_true
            lengths = batch_data[2].to(DEVICE)
            filenames = batch_data[3] if len(batch_data) > 3 else None
        
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
                    
                    # Handle tuple output
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    
                    # Calculate score based on dataset type
                    if processor.dataset_type == 'esc50':
                        # Single-label: score for true class
                        score = logits[0, y_true_clip[i]].item()
                    else:  # audioset
                        true_classes = y_true_clip[i].nonzero(as_tuple=True)[0]
                        false_classes = (y_true_clip[i] == 0).nonzero(as_tuple=True)[0]
                        
                        if len(true_classes) > 0 and len(false_classes) > 0:
                            true_score = logits[0, true_classes].mean()
                            false_score = logits[0, false_classes].mean()
                            
                            # Maximize margin between true and false classes
                            score = (true_score - false_score).item()
                        elif len(true_classes) > 0:
                            score = logits[0, true_classes].mean().item()
                        else:
                            score = logits[0].max().item()
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_wav = wav_compressed.squeeze().cpu()
                    
                    # Update progress bar as we find better scores
                    pbar.set_postfix({
                        'best': f'{best_score:.3f}',
                        'k': k,
                        'ratio': f'{k/F:.2f}'
                    })
            
            # Save best masked audio
            if filenames:
                filename = os.path.basename(filenames[i])
            else:
                filename = f"sample_{i}.wav"
                
            save_path = os.path.join(output_dir, filename)
            torchaudio.save(save_path, best_wav.unsqueeze(0), processor.sr)
            
            # Record results
            results.append({
                'filename': filename,
                'fold_or_split': fold_or_split,
                'kbps': kbps,
                'codec': codec,
                'best_masked_ratio': best_k / F,
                'masked_freq_bins': best_k,
                'total_freq_bins': F,
                'score': best_score,
                'dataset_type': processor.dataset_type
            })
    
    return results

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", default="configs/mask_EnCodec_BEATs_AudioSet.yaml", help="Path to config file")
    ap.add_argument("--output_dir", default="/data/dcase/dataset/masked_audio_output/strong_label_real_f1", help="Output directory")
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="Folds to process (ESC-50 only)")
    ap.add_argument("--kbps_list", nargs="*", type=float, default=[1.5, 3.0, 6.0, 12.0, 24.0], help="kbps levels")
    ap.add_argument("--codec", type=str, default='encodec', choices=['encodec', 'opus'], help="Codec to use")
    ap.add_argument("--step", type=int, default=None, help="Override prune_step from config")
    args = ap.parse_args()
    
    cfg = OmegaConf.load(args.cfgs)
    set_seed(cfg.seed)
    
    # Override step if provided
    if args.step is not None:
        cfg.prune_step = args.step
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    
    all_results = []
    
    # Detect dataset type and process accordingly
    if 'ESC-50' in cfg.data_dir:
        # ESC-50: process by folds
        folds = args.folds if args.folds else [1, 2, 3, 4, 5]
        
        for fold in folds:
            logging.info(f"\n{'='*50}")
            logging.info(f"Processing Fold {fold}")
            logging.info(f"{'='*50}")
            
            processor = AudioProcessor(cfgs=cfg, test_fold=fold)
            saliency_filter = FeatureOnlyFilter(processor)
            
            for kbps in args.kbps_list:
                fold_kbps_dir = os.path.join(args.output_dir, args.codec, f"fold_{fold}", f"{kbps}kbps")
                os.makedirs(fold_kbps_dir, exist_ok=True)
                
                fold_results = process_fold_kbps(
                    processor=processor,
                    saliency_filter=saliency_filter,
                    output_dir=fold_kbps_dir,
                    step=cfg.prune_step,
                    fold_or_split=f"fold_{fold}",
                    kbps=kbps,
                    codec=args.codec
                )
                
                all_results.extend(fold_results)
            
            del processor, saliency_filter
            torch.cuda.empty_cache()
    
    else:  # AudioSet
        logging.info(f"\n{'='*50}")
        logging.info(f"Processing AudioSet Eval Set")
        logging.info(f"{'='*50}")
        
        processor = AudioProcessor(cfgs=cfg)
        saliency_filter = FeatureOnlyFilter(processor)
        
        for kbps in args.kbps_list:
            eval_kbps_dir = os.path.join(args.output_dir, args.codec, f"{kbps}kbps")
            os.makedirs(eval_kbps_dir, exist_ok=True)
            
            eval_results = process_fold_kbps(
                processor=processor,
                saliency_filter=saliency_filter,
                output_dir=eval_kbps_dir,
                step=cfg.prune_step,
                fold_or_split="eval",
                kbps=kbps,
                codec=args.codec
            )
            
            all_results.extend(eval_results)
        
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
    logging.info(f"Saved {len(all_results)} masked audio files")
    logging.info(f"Saved results to {csv_path}")
    
    # Print summary
    logging.info(f"\nOverall Summary:")
    logging.info(f"  Codec: {args.codec.upper()}")
    logging.info(f"  Dataset type: {df['dataset_type'].iloc[0]}")
    logging.info(f"  Mean masking ratio: {df['best_masked_ratio'].mean():.3f}")
    logging.info(f"  Mean masked bins: {df['masked_freq_bins'].mean():.1f}")
    logging.info(f"  Mean score: {df['score'].mean():.3f}")
    
    # Per-split summary
    for split in df['fold_or_split'].unique():
        split_df = df[df['fold_or_split'] == split]
        for kbps in args.kbps_list:
            kbps_df = split_df[split_df['kbps'] == kbps]
            if len(kbps_df) > 0:
                logging.info(f"  {split} @ {kbps}kbps: ratio={kbps_df['best_masked_ratio'].mean():.3f}, score={kbps_df['score'].mean():.3f}")

if __name__ == "__main__":
    main()