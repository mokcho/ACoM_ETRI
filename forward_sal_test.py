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
from sklearn.metrics import accuracy_score, f1_score
from baselines.sound_classification.pipelines import FilterClassifier, EnCodecClassifier
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

import concurrent.futures

# 멀티프로세싱을 위해 클래스 외부에 정의 (Opus 상태 꼬임 방지)
def _process_single_opus(audio_np, sr, bitrate_bps, frame_size, T_wav):
    import opuslib
    import numpy as np
    import torch

    # 파일마다 독립적인 인코더/디코더 객체 생성 (매우 중요)
    encoder = opuslib.Encoder(fs=sr, channels=1, application=opuslib.APPLICATION_AUDIO)
    encoder.bitrate = bitrate_bps
    decoder = opuslib.Decoder(fs=sr, channels=1)

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

    if not decoded_audio:
        return torch.zeros(T_wav, dtype=torch.float32)

    decoded_np = np.concatenate(decoded_audio).astype(np.float32) / 32767.0
    decoded_np = decoded_np[:orig_length]
    
    if len(decoded_np) != T_wav:
        if len(decoded_np) > T_wav:
            decoded_np = decoded_np[:T_wav]
        else:
            decoded_np = np.pad(decoded_np, (0, T_wav - len(decoded_np)), mode='constant')
            
    return torch.from_numpy(decoded_np)

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
            batch_size=self.cfgs.train.batch_size,
            shuffle=False,
            num_workers=self.cfgs.train.num_workers,
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
            batch_size=self.cfgs.train.batch_size,
            shuffle=False,
            num_workers=self.cfgs.train.num_workers,
            collate_fn=collate_fn_multi_class
        )
        
        self.pipeline = BaseClassifier(
            cfgs=self.cfgs, 
            label2id=self.dataset.label2id
        ).to(self.device)
        
        path = self.cfgs.baseline.load_pretrained
        
        # path가 False가 아니고 실제 문자열 경로가 들어왔을 때만 로드
        if path and isinstance(path, str):
            if path.endswith('.pt'):
                load_path = path
            else:
                # 폴더 경로일 경우 해당 폴드의 latest.pt 지정
                load_path = os.path.join(path, f"fold_{test_fold}", "latest.pt")
            
            if os.path.exists(load_path):
                logging.info(f"Loading weights from: {load_path}")
                checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
                self.pipeline.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                logging.warning(f"Weight file not found at {load_path}. Skipping load.")
        else:
            logging.info("No pretrained path string provided. Starting from scratch or base model.")

        
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
            batch_size=self.cfgs.train.batch_size,
            shuffle=False,
            num_workers=self.cfgs.train.num_workers,
            pin_memory=True
        )
        
        if getattr(self.cfgs, 'filters', False):
            self.pipeline = FilterClassifier(cfgs=self.cfgs, label2id=self.dataset.label2id).to(self.device)
        elif getattr(self.cfgs, 'encodec', False):
            self.pipeline = EnCodecClassifier(cfgs=self.cfgs, label2id=self.dataset.label2id).to(self.device)
        else:
            self.pipeline = BaseClassifier(cfgs=self.cfgs, label2id=self.dataset.label2id).to(self.device)
        
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
        
        # ThreadPoolExecutor를 사용하여 배치 내의 오디오를 병렬로 처리
        decoded_batch = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(B, 16)) as executor:
            futures = []
            for i in range(B):
                audio_np = wav16[i, 0].cpu().numpy()
                futures.append(
                    executor.submit(_process_single_opus, audio_np, self.sr, bitrate_bps, frame_size, T_wav)
                )
            
            for future in futures:
                decoded_batch.append(future.result().to(wav16.device))
                
        return torch.stack(decoded_batch, dim=0).unsqueeze(1)

    @torch.no_grad()
    def encodec_roundtrip(self, wav16, kbps):
        self.encodec.set_target_bandwidth(kbps)
        B, _, T_wav = wav16.shape
        if T_wav == 0: return torch.zeros_like(wav16)
        wav24 = AF.resample(wav16, self.sr, 24000)
        enc = self.encodec.encode(wav24)
        y24 = self.encodec.decode(enc)
        enc = self.encodec.encode(y24)
        y24 = self.encodec.decode(enc)
        enc = self.encodec.encode(y24)
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
    # AST: Input Activation Saliency (Forward-only, Pre-Transformer)
    # ------------------------------------------------------------------ #
    def _ast_saliency(self, wav_batch):
        with torch.no_grad():
            S_orig = self.processor.stft(wav_batch)

        pipeline = self.processor.pipeline
        ast_model = pipeline.baseline  # ASTModel 객체
        B, F_target, T_target = S_orig.shape

        # --- Transformer 들어가기 전 layer (patch_embed) 가로채기 ---
        captured_feat = None
        def hook(module, input, output):
            nonlocal captured_feat
            captured_feat = output.detach()
        
        handle = ast_model.v.patch_embed.register_forward_hook(hook)

        with torch.no_grad():
            _ = pipeline(wav_batch, padding_mask=None)
        
        handle.remove()

        if captured_feat is None:
            return S_orig, torch.ones(B, F_target, T_target, device=self.device)

        # BEATs 방식 적용: 특징 벡터의 절대값 평균 (Activation Magnitude)
        token_score = captured_feat.abs().mean(dim=-1)

        maps_list = []
        for b in range(B):
            ts = token_score[b]
            
            f_dim = getattr(ast_model.v, 'f_dim', 12)
            t_dim = ts.shape[0] // f_dim
            
            sal_2d = ts[:f_dim * t_dim].reshape(f_dim, t_dim)
            
            sal = F.interpolate(
                sal_2d.unsqueeze(0).unsqueeze(0), 
                size=(F_target, T_target), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()

            if sal.max() > 0:
                sal = sal / sal.max()

            maps_list.append(sal)

        return S_orig, torch.stack(maps_list).to(self.device)
"""
# ------------------------------
# Processing Functions (🚀 BATCH OPTIMIZED VERSION)
# ------------------------------
def process_fold_kbps(processor, saliency_filter, output_dir, step, fold_or_split, kbps, codec='encodec', target_mode='freq'):
    results = []
    
    logging.info(f"Processing {fold_or_split} @ {kbps} kbps | Codec: {codec.upper()} | Mode: {target_mode.upper()}")
    pbar = tqdm(processor.dataloader, desc=f"{fold_or_split} @ {kbps}k ({target_mode})")
    
    compress_fn = processor.opus_roundtrip if codec == 'opus' else processor.encodec_roundtrip
    first_batch = True
    
    for batch_data in pbar:
        if first_batch:
            first_batch = False
        
        x_orig = batch_data[0].to(DEVICE)  # waveform
        y_true = batch_data[1].to(DEVICE)  # label_id 

        if processor.dataset_type == 'audioset':
            y_true_clip = to_clip_labels(y_true, processor.num_classes)
            filenames = batch_data[3] if len(batch_data) > 3 else None
        else:
            # ESC-50, UrbanSound8K의 경우
            y_true_clip = y_true  # 👈 이 줄이 누락되어 NameError가 났습니다!
            filenames = batch_data[2] if len(batch_data) > 2 else None
        
            
        B = x_orig.shape[0]
        S_orig, sal_map = saliency_filter.get_batch_saliency(x_orig)
        
        if sal_map is None: continue
        
        F_bins, T_bins = sal_map.shape[1], sal_map.shape[2]
        
        if target_mode == 'freq': step_list = list(range(0, int(F_bins * 0.5) + 1, step))
        elif target_mode == 'time': step_list = list(range(0, int(T_bins * 0.5) + 1, step))
        elif target_mode == 'pixel': step_list = [int(F_bins * T_bins * (p / 100.0)) for p in range(0, 51, step)]
        else: step_list = [0]

        # 🚨 루프 순서를 바꿨습니다: 마스킹 단계(k)를 먼저 돌고, 배치 48개를 한 번에 모델에 넣습니다!
        for k in step_list:
            mask_batch = torch.ones((B, F_bins, T_bins), device=DEVICE)
            
            if k > 0:
                for i in range(B):
                    if target_mode == 'freq':
                        scores = sal_map[i].mean(dim=1) 
                        drop_idx = torch.argsort(scores)[:k]
                        mask_batch[i, drop_idx, :] = 0.0
                    elif target_mode == 'time':
                        scores = sal_map[i].mean(dim=0)
                        drop_idx = torch.argsort(scores)[:k]
                        mask_batch[i, :, drop_idx] = 0.0
                    elif target_mode == 'pixel':
                        scores = sal_map[i].flatten()
                        drop_idx = torch.argsort(scores)[:k]
                        mask_batch[i].view(-1)[drop_idx] = 0.0
            
            # [B, F, T] 전체 배치에 한 번에 마스킹 적용
            S_masked_batch = S_orig * mask_batch
            
            # 한 번에 복원 & 압축
            wav_masked_batch = processor.istft(S_masked_batch, length=x_orig.shape[-1])
            wav_c_batch = compress_fn(wav_masked_batch, kbps=kbps)
            
            # 한 번에 48개씩 모델 추론 (GPU 100% 활용)
            with torch.no_grad():
                output = processor.pipeline(wav_c_batch, padding_mask=None)
                logits = output[0] if isinstance(output, tuple) else output
                probs_batch = F.softmax(logits, dim=-1) # [B, Num_Classes]
            
            # 처리된 배치 결과값 정리
            for i in range(B):
                if processor.dataset_type in ['esc50', 'urbansound8k']:
                    true_label = y_true_clip[i].item()
                else:
                    true_classes = y_true_clip[i].nonzero(as_tuple=True)[0]
                    true_label = true_classes[0].item() if len(true_classes) > 0 else -1
                
                if filenames and i < len(filenames) and isinstance(filenames[i], str):
                    filename = os.path.basename(filenames[i])
                else:
                    filename = f"batch_sample_{i}.wav"

                all_probs_list = probs_batch[i].cpu().numpy().tolist()
                true_label_prob = all_probs_list[true_label] if true_label != -1 else 0.0
                
                if target_mode == 'freq': masked_ratio = k / F_bins
                elif target_mode == 'time': masked_ratio = k / T_bins
                else: masked_ratio = k / (F_bins * T_bins)
                
                results.append({
                    'filename': filename,
                    'fold_or_split': fold_or_split,
                    'kbps': kbps,
                    'codec': codec,
                    'mode': target_mode,
                    'step_val': k,
                    'masked_ratio': round(masked_ratio, 3),
                    'true_label': true_label,
                    'true_prob': round(true_label_prob, 5),
                    'all_probs': [round(p, 5) for p in all_probs_list]
                })
                
    return results
"""

# ------------------------------
# Processing Functions (🚀 BATCH OPTIMIZED VERSION)
# ------------------------------
def process_fold_kbps(processor, saliency_filter, output_dir, step, fold_or_split, kbps, codec='encodec', target_mode='freq'):
    from sklearn.metrics import accuracy_score, f1_score
    results = []
    
    logging.info(f"Processing {fold_or_split} @ {kbps} kbps | Codec: {codec.upper()} | Mode: {target_mode.upper()}")
    pbar = tqdm(processor.dataloader, desc=f"{fold_or_split} @ {kbps}k ({target_mode})")
    
    # 밖에서 강제로 정직하게 압축하는 함수 세팅
    compress_fn = processor.opus_roundtrip if codec == 'opus' else processor.encodec_roundtrip
    first_batch = True
    
    for batch_data in pbar:
        if first_batch:
            first_batch = False
        
        x_orig = batch_data[0].to(DEVICE)  
        y_true = batch_data[1].to(DEVICE)  

        if processor.dataset_type == 'audioset':
            y_true_clip = to_clip_labels(y_true, processor.num_classes)
            filenames = batch_data[3] if len(batch_data) > 3 else None
        else:
            y_true_clip = y_true  
            filenames = batch_data[2] if len(batch_data) > 2 else None
            
        B = x_orig.shape[0]
        S_orig, sal_map = saliency_filter.get_batch_saliency(x_orig)
        
        if sal_map is None: continue
        
        F_bins, T_bins = sal_map.shape[1], sal_map.shape[2]
        
        if step == 0: 
            step_list = [0]
        else:
            if target_mode == 'freq': step_list = list(range(0, int(F_bins * 0.5) + 1, step))
            elif target_mode == 'time': step_list = list(range(0, int(T_bins * 0.5) + 1, step))
            elif target_mode == 'pixel': step_list = [int(F_bins * T_bins * (p / 100.0)) for p in range(0, 51, step)]
            else: step_list = [0]

        for k in step_list:
            mask_batch = torch.ones((B, F_bins, T_bins), device=DEVICE)
            
            # 🔥 핵심: AST, BEATs, Opus, Encodec 가릴 것 없이 무조건 "외부 수동 강제 압축" 적용
            if k == 0:
                wav_c_batch = compress_fn(x_orig, kbps=kbps)
            else:
                for i in range(B):
                    if target_mode == 'freq':
                        scores = sal_map[i].mean(dim=1) 
                        drop_idx = torch.argsort(scores)[:k]
                        mask_batch[i, drop_idx, :] = 0.0
                    elif target_mode == 'time':
                        scores = sal_map[i].mean(dim=0)
                        drop_idx = torch.argsort(scores)[:k]
                        mask_batch[i, :, drop_idx] = 0.0
                    elif target_mode == 'pixel':
                        scores = sal_map[i].flatten()
                        drop_idx = torch.argsort(scores)[:k]
                        mask_batch[i].view(-1)[drop_idx] = 0.0
                
                S_masked_batch = S_orig * mask_batch
                wav_masked_batch = processor.istft(S_masked_batch, length=x_orig.shape[-1])
                wav_c_batch = compress_fn(wav_masked_batch, kbps=kbps)
            
            with torch.no_grad():
                # 내부 Encodec 의존 없이 무조건 Base 모델에 압축된 파형을 집어넣음
                output = processor.pipeline(wav_c_batch, eval_mode=True)
                
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                    
                probs_batch = F.softmax(logits, dim=-1) 
            
            for i in range(B):
                if processor.dataset_type in ['esc50', 'urbansound8k']:
                    true_label = y_true_clip[i].item()
                else:
                    true_classes = y_true_clip[i].nonzero(as_tuple=True)[0]
                    true_label = true_classes[0].item() if len(true_classes) > 0 else -1
                
                if filenames and i < len(filenames) and isinstance(filenames[i], str):
                    filename = os.path.basename(filenames[i])
                else:
                    filename = f"batch_sample_{i}.wav"

                all_probs_list = probs_batch[i].cpu().numpy().tolist()
                true_label_prob = all_probs_list[true_label] if true_label != -1 else 0.0
                
                if target_mode == 'freq': masked_ratio = k / F_bins
                elif target_mode == 'time': masked_ratio = k / T_bins
                else: masked_ratio = k / (F_bins * T_bins)
                
                results.append({
                    'filename': filename,
                    'fold_or_split': fold_or_split,
                    'kbps': kbps,
                    'codec': codec,
                    'mode': target_mode,
                    'step_val': k,
                    'masked_ratio': round(masked_ratio, 3),
                    'true_label': true_label,
                    'true_prob': round(true_label_prob, 5),
                    'all_probs': [round(p, 5) for p in all_probs_list]
                })

    if len(results) > 0:
        y_true = [r['true_label'] for r in results if r['step_val'] == 0]
        y_pred = [np.argmax(r['all_probs']) for r in results if r['step_val'] == 0]
        
        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            logging.info(f"\n[결과] {fold_or_split} @ {kbps}kbps ({codec.upper()} 진짜 1.5kbps 압축 적용) -> Accuracy: {acc:.4f}, F1: {f1_macro:.4f}\n")
                
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", default="configs/test_BEATs_UrbanSound8K.yaml")
    ap.add_argument("--output_dir", default="/data/dcase/dataset/masked_audio_output/strong_label_real_f1")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--kbps_list", nargs="*", type=float, default=[1.5, 3.0, 6.0, 12.0, 24.0])
    ap.add_argument("--codec", type=str, default='encodec', choices=['encodec', 'opus'])
    ap.add_argument("--step", type=int, default=10) # None에 대비해 기본값 10 설정
    # 🚨 모드 선택 옵션
    ap.add_argument("--mode", type=str, default='freq', choices=['freq', 'time', 'pixel'])
    args = ap.parse_args()
    
    cfg = OmegaConf.load(args.cfgs)
    set_seed(cfg.seed)
    
    if args.step is not None:
        cfg.prune_step = args.step
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    
    model_type = cfg.baseline.model.lower()
    logging.info(f"Model: {model_type.upper()} | Codec: {args.codec.upper()} | Mode: {args.mode.upper()}")
    
    all_results = []
    
    if 'ESC-50' in cfg.data_dir:
        folds = args.folds if args.folds else [1, 2, 3, 4, 5]
        
        for fold in folds:
            logging.info(f"\n{'='*50}\nProcessing Fold {fold} [{model_type.upper()}]\n{'='*50}")
            processor = AudioProcessor(cfgs=cfg, test_fold=fold)
            saliency_filter = FeatureOnlyFilter(processor)
            
            for kbps in args.kbps_list:
                fold_label = f"fold_{fold}" # 🔥 변수를 명시적으로 선언
                fold_kbps_dir = os.path.join(args.output_dir, args.codec, f"{kbps}kbps", fold_label)
                os.makedirs(fold_kbps_dir, exist_ok=True)
                fold_results = process_fold_kbps(
                    processor=processor, saliency_filter=saliency_filter,
                    output_dir=fold_kbps_dir, step=cfg.prune_step,
                    fold_or_split=fold_label, # 🔥 선언한 변수 사용
                    kbps=kbps, codec=args.codec,
                    target_mode=args.mode 
                )

                if fold_results:
                    df_temp = pd.DataFrame(fold_results)
                    # 🔥 fold_label(예: fold_1)이 파일명에 정확히 박힙니다.
                    temp_csv_name = f"results_{model_type}_{fold_label}_{kbps}kbps_{args.mode}.csv"
                    temp_csv_path = os.path.join(fold_kbps_dir, temp_csv_name)
                    df_temp.to_csv(temp_csv_path, index=False)
                all_results.extend(fold_results)
            del processor, saliency_filter
            torch.cuda.empty_cache()

    elif 'urbansound8k' in cfg.data_dir.lower():
        folds = args.folds if args.folds else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for fold in folds:
            logging.info(f"\n{'='*50}\nProcessing Fold {fold} [{model_type.upper()}]\n{'='*50}")
            processor = AudioProcessor(cfgs=cfg, test_fold=fold)
            saliency_filter = FeatureOnlyFilter(processor)
            
            for kbps in args.kbps_list:
                fold_label = f"fold_{fold}" # 🔥 변수를 명시적으로 선언
                fold_kbps_dir = os.path.join(args.output_dir, args.codec, f"{kbps}kbps", fold_label)
                os.makedirs(fold_kbps_dir, exist_ok=True)
                fold_results = process_fold_kbps(
                    processor=processor, saliency_filter=saliency_filter,
                    output_dir=fold_kbps_dir, step=cfg.prune_step,
                    fold_or_split=fold_label, # 🔥 선언한 변수 사용
                    kbps=kbps, codec=args.codec,
                    target_mode=args.mode 
                )

                if fold_results:
                    df_temp = pd.DataFrame(fold_results)
                    # 🔥 fold_label(예: fold_1)이 파일명에 정확히 박힙니다.
                    temp_csv_name = f"results_{model_type}_{fold_label}_{kbps}kbps_{args.mode}.csv"
                    temp_csv_path = os.path.join(fold_kbps_dir, temp_csv_name)
                    df_temp.to_csv(temp_csv_path, index=False)
                all_results.extend(fold_results)
            del processor, saliency_filter
            torch.cuda.empty_cache()
    
    else:  # AudioSet
        logging.info(f"\n{'='*50}\nProcessing AudioSet Eval Set [{model_type.upper()}]\n{'='*50}")
        processor = AudioProcessor(cfgs=cfg)
        saliency_filter = FeatureOnlyFilter(processor)
        
        for kbps in args.kbps_list:
            eval_kbps_dir = os.path.join(args.output_dir, args.codec, f"{kbps}kbps")
            os.makedirs(eval_kbps_dir, exist_ok=True)
            
            # 🚨 target_mode=args.mode 추가!
            eval_results = process_fold_kbps(
                processor=processor, saliency_filter=saliency_filter,
                output_dir=eval_kbps_dir, step=cfg.prune_step,
                fold_or_split="eval", kbps=kbps, codec=args.codec,
                target_mode=args.mode
            )
            all_results.extend(eval_results)
        del processor, saliency_filter
        torch.cuda.empty_cache()
    
    # Save combined CSV
    if all_results:
        df = pd.DataFrame(all_results)
        # 파일 이름에 mode 추가해서 나중에 헷갈리지 않게 변경 (예: masking_results_time.csv)
        csv_path = os.path.join(args.output_dir, args.codec, f'masking_results_{args.mode}.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        logging.info(f"\n{'='*50}\nALL PROCESSING COMPLETE\n{'='*50}")
        logging.info(f"Model: {model_type.upper()} | Codec: {args.codec.upper()} | Mode: {args.mode.upper()}")
        logging.info(f"Saved {len(all_results)} rows to {csv_path}")

if __name__ == "__main__":
    main()