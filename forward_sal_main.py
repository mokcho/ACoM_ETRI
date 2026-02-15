import os, glob, argparse, logging, math, re
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

from baselines.sound_classification.pipelines import BaseClassifier
from data import ESCDataset, collate_fn_multi_class, UrbanSound8KDataset


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
    if labels.shape[1] == num_classes:
        clip = labels.amax(dim=2)
    else:
        clip = labels.transpose(1, 2).amax(dim=2)
    return clip

def update_data_dir(data_dir, fold=None, kbps=None):
    """Replace fold_N and kbps_X.X in data_dir path"""
    if fold is not None:
        data_dir = re.sub(r'fold_\d+', f'fold_{fold}', data_dir)
    if kbps is not None:
        data_dir = re.sub(r'kbps_[\d.]+', f'kbps_{kbps}', data_dir)
    return data_dir

# ------------------------------
# Unified AudioProcessor
# ------------------------------
class AudioProcessor:
    def __init__(self, cfgs, test_fold=None):
        self.device = DEVICE
        self.cfgs = cfgs
        self.model_type = cfgs.baseline.model.lower()
        self.dataset_type = self._detect_dataset_type()
        
        if self.dataset_type == 'esc50':
            self._init_esc50(test_fold)
        elif self.dataset_type == 'urbansound8k':
            self._init_urbansound8k(test_fold)
        elif self.dataset_type == 'audioset':
            self._init_audioset()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        
        # Common STFT components
        n_fft = getattr(self.cfgs.data, 'n_fft', 1024)
        hop = getattr(self.cfgs.data, 'hop', 256)
        
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop, power=None, return_complex=True
        ).to(self.device)
        self.ispec = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft, hop_length=hop
        ).to(self.device)
        
        self.encodec = EncodecModel.encodec_model_24khz().to(self.device).eval()
        self._opus_encoder_cache = {}
        self._opus_decoder_cache = {}
    
    def _detect_dataset_type(self):
        if 'ESC-50' in self.cfgs.data_dir:
            return 'esc50'
        elif 'urbansound8k' in self.cfgs.data_dir.lower():
            return 'urbansound8k'
        elif 'dcase' in self.cfgs.data_dir or 'audioset' in self.cfgs.annotation_dir.lower() or hasattr(self.cfgs, 'mode'):
            return 'audioset'
        else:
            raise ValueError("Cannot detect dataset type from config")
    
    def _init_esc50(self, test_fold):
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
        
        path = self.cfgs.baseline.load_pretrained
        logging.info(f"Loading ESC-50 {self.model_type.upper()} model from fold {test_fold}")
        
        if path.endswith('.pt'):
            checkpoint = torch.load(path, map_location=self.device)
        else:
            checkpoint = torch.load(
                os.path.join(path, f"fold_{test_fold}", "latest.pt"), 
                map_location=self.device
            )
        
        self.pipeline.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.pipeline.eval()
        
        # BEATs-specific attributes (only set if model is BEATs)
        if self.model_type == 'beats':
            self.hook_module = self.pipeline.baseline.layer_norm
            self.beats_cfg = self.pipeline.baseline.cfg

    def _init_urbansound8k(self, test_fold):
        self.sr = self.cfgs.data.sr
        self.num_classes = 10
        
        self.dataset = UrbanSound8KDataset(
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
            collate_fn=collate_fn_multi_class
        )
        
        self.pipeline = BaseClassifier(
            cfgs=self.cfgs, 
            label2id=self.dataset.label2id
        ).to(self.device)
        
        path = self.cfgs.baseline.load_pretrained
        
        if path and isinstance(path, str):
            if path.endswith('.pt'):
                load_path = path
            else:
                load_path = os.path.join(path, f"fold_{test_fold}", "latest.pt")
            
            if os.path.exists(load_path):
                logging.info(f"Loading weights from: {load_path}")
                checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
                self.pipeline.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                logging.warning(f"Weight file not found at {load_path}. Skipping load.")
        else:
            raise ValueError("A pretrained model path must be provided.")
        
        self.pipeline.eval()
        
        if self.model_type == 'beats':
            self.hook_module = self.pipeline.baseline.layer_norm
            self.beats_cfg = self.pipeline.baseline.cfg
            
    def _init_audioset(self):
        self.sr = self.cfgs.data.sr
        
        eval_annotation = self.cfgs.annotation_dir.replace("train", "eval")
        val_tsv = pd.read_csv(eval_annotation, sep="\t")
        
        self.encoder = ManyHotEncoder(
            labels=self.cfgs.data.classes,
            n_frames=self.cfgs.data.n_frames
        )
        
        self.num_classes = len(self.cfgs.data.classes)
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
        
        if hasattr(self.cfgs, 'filters') and self.cfgs.filters:
            self.pipeline = FilterEventDetector(
                self.cfgs, num_classes=self.num_classes, mode=self.cfgs.mode
            ).to(self.device)
        elif hasattr(self.cfgs, 'encodec') and self.cfgs.encodec:
            self.pipeline = EnCodecEventDetector(
                self.cfgs, num_classes=self.num_classes, mode=self.cfgs.mode
            ).to(self.device)
        else:
            self.pipeline = BaseEventDetector(
                self.cfgs, num_classes=self.num_classes, mode=self.cfgs.mode
            ).to(self.device)
        
        path = self.cfgs.baseline.load_pretrained
        logging.info(f"Loading AudioSet {self.model_type.upper()} model from {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.pipeline.load_state_dict(checkpoint['strong_model_state_dict'], strict=False)
        self.pipeline.eval()
        
        # BEATs-specific attributes
        if self.model_type == 'beats':
            self.hook_module = self.pipeline.baseline.layer_norm
            self.beats_cfg = self.pipeline.baseline.cfg

    def stft(self, wav): 
        return self.spec(wav.squeeze(1))
    
    def istft(self, S, length): 
        return self.ispec(S, length=length).unsqueeze(1)
    
    def _get_opus_encoder(self, bitrate_bps):
        if bitrate_bps not in self._opus_encoder_cache:
            encoder = opuslib.Encoder(fs=self.sr, channels=1, application=opuslib.APPLICATION_AUDIO)
            encoder.bitrate = bitrate_bps
            self._opus_encoder_cache[bitrate_bps] = encoder
        return self._opus_encoder_cache[bitrate_bps]
    
    def _get_opus_decoder(self):
        if 'decoder' not in self._opus_decoder_cache:
            self._opus_decoder_cache['decoder'] = opuslib.Decoder(fs=self.sr, channels=1)
        return self._opus_decoder_cache['decoder']
    
    @torch.no_grad()
    def opus_roundtrip(self, wav16, kbps):
        B, _, T_wav = wav16.shape
        if T_wav == 0:
            return torch.zeros_like(wav16)
        
        bitrate_bps = int(kbps * 1000)
        frame_size = int(self.sr * 0.02)
        encoder = self._get_opus_encoder(bitrate_bps)
        decoder = self._get_opus_decoder()
        
        decoded_batch = []
        for i in range(B):
            audio_np = wav16[i, 0].cpu().numpy()
            orig_length = len(audio_np)
            pad_length = (frame_size - (orig_length % frame_size)) % frame_size
            if pad_length > 0:
                audio_np = np.pad(audio_np, (0, pad_length), mode='constant')
            audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
            
            encoded_frames = []
            for j in range(0, len(audio_int16), frame_size):
                frame = audio_int16[j:j+frame_size]
                if len(frame) == frame_size:
                    encoded = encoder.encode(frame.tobytes(), frame_size)
                    encoded_frames.append(encoded)
            
            decoded_audio = []
            for encoded in encoded_frames:
                decoded = decoder.decode(encoded, frame_size)
                decoded_int16 = np.frombuffer(decoded, dtype=np.int16)
                decoded_audio.append(decoded_int16)
            
            decoded_np = np.concatenate(decoded_audio).astype(np.float32) / 32767.0
            decoded_np = decoded_np[:orig_length]
            if len(decoded_np) != T_wav:
                if len(decoded_np) > T_wav:
                    decoded_np = decoded_np[:T_wav]
                else:
                    decoded_np = np.pad(decoded_np, (0, T_wav - len(decoded_np)), mode='constant')
            decoded_batch.append(torch.from_numpy(decoded_np).to(wav16.device))
        
        return torch.stack(decoded_batch, dim=0).unsqueeze(1)
    
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
# Saliency Filter (BEATs + AST)
# ------------------------------
class FeatureOnlyFilter:
    """
    Dispatches saliency computation based on model type:
      - BEATs: activation magnitude from layer_norm hook (forward-only)
      - AST:   attention rollout from CLS token (forward-only, no backprop)
    """
    def __init__(self, processor):
        self.processor = processor
        self.device = processor.device
        self.model_type = processor.model_type

    def get_batch_saliency(self, wav_batch):
        if self.model_type == 'beats':
            return self._beats_saliency(wav_batch)
        elif self.model_type == 'ast':
            return self._ast_saliency(wav_batch)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    # ------------------------------------------------------------------ #
    # BEATs: activation magnitude from layer_norm (original approach)
    # ------------------------------------------------------------------ #
    def _beats_saliency(self, wav_batch):
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
            grid = ts[:F_prime * N_prime].view(F_prime, N_prime)
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

    # ------------------------------------------------------------------ #
    # AST: attention rollout from CLS token (forward-only, no backprop)
    # ------------------------------------------------------------------ #
    def _ast_saliency(self, wav_batch):
        with torch.no_grad():
            S_orig = self.processor.stft(wav_batch)

        pipeline = self.processor.pipeline
        ast_model = pipeline.baseline  # ASTModel
        B, F_target, T_target = S_orig.shape

        # --- Register hooks on attn_drop to capture attention weights ---
        attn_weights_per_layer = []
        hooks = []

        def make_attn_hook(storage):
            def hook_fn(module, inp, output):
                # inp[0] = attention weights after softmax: (B, num_heads, N, N)
                storage.append(inp[0].detach())
            return hook_fn

        attn_drop_modules = self._find_attn_drop_modules(ast_model)
        
        if not attn_drop_modules:
            logging.warning("Could not find attn_drop modules in AST model. "
                          "Falling back to uniform saliency.")
            sal_batch = torch.ones(B, F_target, T_target, device=self.device)
            return S_orig, sal_batch

        for mod in attn_drop_modules:
            hooks.append(mod.register_forward_hook(make_attn_hook(attn_weights_per_layer)))

        # --- Forward pass (pipeline handles mel preprocessing for AST) ---
        with torch.no_grad():
            _ = pipeline(wav_batch, padding_mask=None)

        for h in hooks:
            h.remove()

        if not attn_weights_per_layer:
            logging.warning("No attention weights captured. Check AST model structure.")
            sal_batch = torch.ones(B, F_target, T_target, device=self.device)
            return S_orig, sal_batch

        # --- Attention rollout (per sample in batch) ---
        maps_list = []
        
        for b in range(B):
            n_tokens = attn_weights_per_layer[0].shape[-1]
            rollout = torch.eye(n_tokens, device=self.device)

            for attn in attn_weights_per_layer:
                a = attn[b]  # (heads, N, N) or (N, N)
                if a.dim() == 3:
                    a = a.mean(dim=0)  # average over heads -> (N, N)
                # Add residual connection (skip connection)
                a = 0.5 * a + 0.5 * torch.eye(n_tokens, device=self.device)
                a = a / a.sum(dim=-1, keepdim=True)
                rollout = a @ rollout

            # CLS token (index 0) attention to patch tokens
            # AST DeiT has CLS + distillation token at indices 0, 1
            cls_attn = rollout[0, 2:]  # (num_patches,)

            # Reshape to 2D patch grid
            n_patches = cls_attn.shape[0]
            f_dim, t_dim = self._get_ast_grid_dims(ast_model, n_patches)
            
            usable = f_dim * t_dim
            sal_2d = cls_attn[:usable].reshape(f_dim, t_dim)

            # Interpolate to STFT size (F_target, T_target)
            sal = F.interpolate(
                sal_2d.unsqueeze(0).unsqueeze(0),
                size=(F_target, T_target),
                mode='bilinear',
                align_corners=False
            ).squeeze()

            if sal.max() > 0:
                sal = sal / sal.max()
            maps_list.append(sal)

        sal_batch = torch.stack(maps_list).to(self.device)
        return S_orig, sal_batch

    def _find_attn_drop_modules(self, ast_model):
        """
        Locate attn_drop (nn.Dropout) modules inside timm 0.4.5 Attention blocks.
        In timm's Attention.forward():
            attn = (q @ k.T) * scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)    <-- we hook HERE
            x = (attn @ v)
        """
        modules = []
        
        # Path 1: ast_model.v.blocks[i].attn.attn_drop (standard timm DeiT)
        if hasattr(ast_model, 'v') and hasattr(ast_model.v, 'blocks'):
            for block in ast_model.v.blocks:
                if hasattr(block, 'attn') and hasattr(block.attn, 'attn_drop'):
                    modules.append(block.attn.attn_drop)
            if modules:
                return modules

        # Path 2: ast_model.blocks[i].attn.attn_drop
        if hasattr(ast_model, 'blocks'):
            for block in ast_model.blocks:
                if hasattr(block, 'attn') and hasattr(block.attn, 'attn_drop'):
                    modules.append(block.attn.attn_drop)
            if modules:
                return modules

        # Path 3: brute-force — find all Dropout named 'attn_drop'
        for name, mod in ast_model.named_modules():
            if 'attn_drop' in name and isinstance(mod, nn.Dropout):
                modules.append(mod)
        
        return modules

    def _get_ast_grid_dims(self, ast_model, n_patches):
        """Get (f_dim, t_dim) patch grid dimensions from AST model."""
        for container in [ast_model, getattr(ast_model, 'v', None)]:
            if container is not None:
                f_dim = getattr(container, 'f_dim', None)
                t_dim = getattr(container, 't_dim', None)
                if f_dim is not None and t_dim is not None:
                    if f_dim * t_dim <= n_patches:
                        return int(f_dim), int(t_dim)

        # Fallback: try common f_dim values for 128-mel AST
        for f_guess in [12, 8, 13, 16]:
            if n_patches % f_guess == 0:
                return f_guess, n_patches // f_guess

        # Last resort: find closest factor pair
        t_dim = int(math.sqrt(n_patches))
        while t_dim > 0 and n_patches % t_dim != 0:
            t_dim -= 1
        f_dim = n_patches // t_dim if t_dim > 0 else n_patches
        return f_dim, t_dim

# ------------------------------
# Processing Functions
# ------------------------------
def process_fold_kbps(processor, saliency_filter, output_dir, step, fold_or_split, kbps, codec='encodec'):
    results = []
    
    logging.info(f"Processing {fold_or_split} @ {kbps} kbps with {codec.upper()} [{processor.model_type.upper()}]")
    
    pbar = tqdm(processor.dataloader, desc=f"{fold_or_split} @ {kbps}kbps ({codec})")
    
    if codec == 'opus':
        compress_fn = processor.opus_roundtrip
    else:
        compress_fn = processor.encodec_roundtrip
    
    first_batch = True
    
    for batch_data in pbar:
        
        if first_batch:
            logging.info(f"Batch length: {len(batch_data)}")
            if len(batch_data) > 3:
                logging.info(f"Filenames found: {batch_data[3][0] if batch_data[3] else 'None'}")
            elif isinstance(batch_data[2], list) and batch_data[2] and isinstance(batch_data[2][0], str):
                logging.info(f"Filenames found: {batch_data[2][0]}")
            else:
                logging.warning("Filenames NOT in batch! Files will be saved as sample_N.wav")
            first_batch = False
        
        x_orig = batch_data[0].to(DEVICE)
        y_true = batch_data[1].to(DEVICE)
        
        if processor.dataset_type == 'audioset':
            y_true_clip = to_clip_labels(y_true, processor.num_classes)
            filenames = batch_data[3] if len(batch_data) > 3 else None
            lengths = torch.full((x_orig.shape[0],), x_orig.shape[-1], device=DEVICE, dtype=torch.long)

        else:  # ESC-50 or UrbanSound8K
            y_true_clip = y_true
            if isinstance(batch_data[2], list): # UrbanSound8K
                filenames = batch_data[2]
                lengths = torch.full((x_orig.shape[0],), x_orig.shape[-1], device=DEVICE, dtype=torch.long)
            else: # ESC-50 
                lengths = batch_data[2].to(DEVICE)
                filenames = batch_data[3] if len(batch_data) > 3 else None
        
        B = x_orig.shape[0]
        
        S_orig, sal_map = saliency_filter.get_batch_saliency(x_orig)
        
        if sal_map is None:
            logging.warning("Saliency computation failed, skipping batch")
            continue
        
        freq_scores = sal_map.mean(dim=2)  # (B, F)
        F_bins = freq_scores.shape[1]
        max_remove = int(F_bins * 0.5)
        
        for i in range(B):
            best_score = -1e9
            best_k = 0
            best_wav = None
            
            for k in range(0, max_remove + 1, step):
                mask = torch.ones((F_bins, 1), device=DEVICE)
                if k > 0:
                    sorted_idx = torch.argsort(freq_scores[i], descending=False)
                    low_k_idx = sorted_idx[:k]
                    mask[low_k_idx] = 0.0
                
                S_masked = S_orig[i] * mask
                wav = processor.istft(S_masked.unsqueeze(0), length=lengths[i])
                wav_compressed = compress_fn(wav, kbps=kbps)
                
                with torch.no_grad():
                    output = processor.pipeline(wav_compressed, padding_mask=None)
                    
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    
                    if processor.dataset_type == 'esc50':
                        score = logits[0, y_true_clip[i]].item()
                    else:  # audioset
                        true_classes = y_true_clip[i].nonzero(as_tuple=True)[0]
                        false_classes = (y_true_clip[i] == 0).nonzero(as_tuple=True)[0]
                        
                        if len(true_classes) > 0 and len(false_classes) > 0:
                            true_score = logits[0, true_classes].mean()
                            false_score = logits[0, false_classes].mean()
                            score = (true_score - false_score).item()
                        elif len(true_classes) > 0:
                            score = logits[0, true_classes].mean().item()
                        else:
                            score = logits[0].max().item()
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_wav = wav_compressed.squeeze().cpu()
                    
                    pbar.set_postfix({
                        'best': f'{best_score:.3f}',
                        'k': k,
                        'ratio': f'{k/F_bins:.2f}'
                    })
            
            #if filenames:
            #    filename = os.path.basename(filenames[i])
            #else:
            #    filename = f"sample_{i}.wav"

            if filenames and i < len(filenames) and isinstance(filenames[i], str):
                filename = os.path.basename(filenames[i])
                if not filename.endswith('.wav'):
                    filename = filename + '.wav'
            else:
                filename = f"batch_sample_{i}.wav"
                
            save_path = os.path.join(output_dir, filename)
            torchaudio.save(save_path, best_wav.unsqueeze(0), processor.sr)
            
            results.append({
                'filename': filename,
                'fold_or_split': fold_or_split,
                'kbps': kbps,
                'codec': codec,
                'model': processor.model_type,
                'best_masked_ratio': best_k / F_bins,
                'masked_freq_bins': best_k,
                'total_freq_bins': F_bins,
                'score': best_score,
                'dataset_type': processor.dataset_type
            })
    
    return results

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", default="configs/mask_EnCodec_BEATs_AudioSet.yaml")
    ap.add_argument("--output_dir", default="/data/dcase/dataset/masked_audio_output/strong_label_real_f1")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--kbps_list", nargs="*", type=float, default=[1.5, 3.0, 6.0, 12.0, 24.0])
    ap.add_argument("--codec", type=str, default='encodec', choices=['encodec', 'opus'])
    ap.add_argument("--step", type=int, default=None)
    args = ap.parse_args()
    
    cfg = OmegaConf.load(args.cfgs)
    set_seed(cfg.seed)
    
    if args.step is not None:
        cfg.prune_step = args.step
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    
    model_type = cfg.baseline.model.lower()
    logging.info(f"Model: {model_type.upper()} | Codec: {args.codec.upper()}")
    
    all_results = []
    
    if 'ESC-50' in cfg.data_dir:
        folds = args.folds if args.folds else [1, 2, 3, 4, 5]
        
        for fold in folds:
            logging.info(f"\n{'='*50}")
            logging.info(f"Processing Fold {fold} [{model_type.upper()}]")
            logging.info(f"{'='*50}")
            
            processor = AudioProcessor(cfgs=cfg, test_fold=fold)
            saliency_filter = FeatureOnlyFilter(processor)
            
            for kbps in args.kbps_list:
                fold_kbps_dir = os.path.join(args.output_dir, args.codec, f"fold_{fold}", f"kbps_{kbps}")
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

    elif 'urbansound8k' in cfg.data_dir.lower():
        folds = args.folds if args.folds else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for fold in folds:
            logging.info(f"\n{'='*50}")
            logging.info(f"Processing Fold {fold} [{model_type.upper()}]")
            logging.info(f"{'='*50}")
            
            processor = AudioProcessor(cfgs=cfg, test_fold=fold)
            saliency_filter = FeatureOnlyFilter(processor)
            
            for kbps in args.kbps_list:
                fold_kbps_dir = os.path.join(args.output_dir, args.codec, f"fold_{fold}", f"kbps_{kbps}")
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
        logging.info(f"Processing AudioSet Eval Set [{model_type.upper()}]")
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
    logging.info(f"Model: {model_type.upper()} | Codec: {args.codec.upper()}")
    logging.info(f"Saved {len(all_results)} masked audio files")
    logging.info(f"Saved results to {csv_path}")
    
    logging.info(f"\nOverall Summary:")
    logging.info(f"  Mean masking ratio: {df['best_masked_ratio'].mean():.3f}")
    logging.info(f"  Mean masked bins: {df['masked_freq_bins'].mean():.1f}")
    logging.info(f"  Mean score: {df['score'].mean():.3f}")
    
    for split in df['fold_or_split'].unique():
        split_df = df[df['fold_or_split'] == split]
        for kbps in args.kbps_list:
            kbps_df = split_df[split_df['kbps'] == kbps]
            if len(kbps_df) > 0:
                logging.info(f"  {split} @ {kbps}kbps: ratio={kbps_df['best_masked_ratio'].mean():.3f}, score={kbps_df['score'].mean():.3f}")

if __name__ == "__main__":
    main()