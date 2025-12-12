import os
import glob
import argparse

from collections import defaultdict
import numpy as np
import pandas as pd
import json

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF

from types import SimpleNamespace
from baselines.beats.BEATs import BEATs, BEATsConfig
from encodec import EncodecModel

# ------------------------------
# 1. 설정 및 기본 클래스
# ------------------------------
SR =16000
N_FFT =1024
HOP = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BEATsClassifier(nn.Module):
    def __init__(self, beats_ckpt_path, num_classes, freeze_beats=True):
        super().__init__()
        ckpt = torch.load(beats_ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]; cfg["finetuned_model"] = False
        self.beats = BEATs(BEATsConfig(cfg))
        self.beats.load_state_dict(ckpt["model"], strict=False)
        if freeze_beats:
            for p in self.beats.parameters():
                p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(self.beats.cfg.encoder_embed_dim, 256),
            nn.Dropout(0.2), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(1)
        feats = self.beats.extract_features(x, padding_mask=None)[0]
        return self.classifier(feats.mean(dim=1))

class ESC50Info:
    def __init__(self, csv_path):
        meta = pd.read_csv(csv_path)
        self.target2cat = dict(zip(meta['target'].values, meta['category'].values))
        cats = sorted(meta['category'].unique())
        self.label2id = {c: i for i, c in enumerate(cats)}
        self.id2label = {i: c for c, i in self.label2id.items()}

    def label_from_filename(self, fn: str):
        try:
            target = int(fn.split('-')[-1].replace('.wav', ''))
            cat = self.target2cat[target]
            return self.label2id[cat]
        except:
            return None

class AudioProcessor:
    def __init__(self, cfg, esc50_csv, beats_ckpt_path, clf_path=None, sr=SR, n_fft=N_FFT, hop=HOP):
        self.device = DEVICE
        self.clf_path = clf_path
        self.sr = sr
        self.esc = ESC50Info(esc50_csv)
        self.num_classes = len(self.esc.label2id)
        self.clf = BEATsClassifier(beats_ckpt_path, self.num_classes, freeze_beats=True).to(self.device)
        if clf_path and os.path.isfile(clf_path):
            sd = torch.load(clf_path, map_location=self.device, weights_only=False)
            state = sd.get('model_state_dict', sd)
            head = {k[len('classifier.'):]: v for k, v in state.items() if k.startswith('classifier.')}
            self.clf.classifier.load_state_dict(head, strict=True)
            print(f"Loaded classifier head from {clf_path}")
        else:
            print("Warning: classifier head not loaded; using random init.")
            
        
        self.clf.eval() # 평가 모드로 고정
        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop, power=None, return_complex=True).to(self.device)
        self.ispec = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, hop_length=hop).to(self.device)
        
        if cfg.kbps > 0:
            self.encodec = EncodecModel.encodec_model_24khz().to(self.device).eval()
            

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
        return wav16

    def stft(self, wav): return self.spec(wav.squeeze(1))
    def istft(self, S, length): return self.ispec(S, length=length).unsqueeze(1)

    @torch.no_grad()
    def classify(self, wav): return self.clf(wav)

# ------------------------------
# 2. 보조 함수 
# ------------------------------
   
def find_optimal_global_pruning(processor: AudioProcessor, x_orig, S_orig, T_wav, saliency_map_resized, y_true, kbps):
    """
    Saliency를 시간 축에 대해 평균내어 '전역 주파수 중요도'를 계산합니다.
    중요도가 낮은 주파수를 순차적으로 제거하며 최적의 필터를 찾습니다.
    """
    # 1. Saliency map을 시간 축(dim=1)에 대해 평균내어 전역 중요도 계산
    global_saliency_scores = torch.mean(saliency_map_resized, dim=1)
    
    # 2. 중요도가 낮은 순서(오름차순)대로 주파수 인덱스를 정렬
    sorted_indices = torch.argsort(global_saliency_scores, descending=False)
    
    n_freqs = S_orig.shape[1]
    best_logit = -float('inf')
    best_rebuilt_audio = None
    best_s_rebuilt = None
    best_num_removed = -1 # 초기값을 -1로 설정하여 아무것도 선택되지 않는 경우를 확인

    # --- 원본(0개 제거)에 대한 평가를 먼저 수행 ---
    with torch.no_grad():
        if kbps > 0 :
            x_filtered_orig = processor.encodec_roundtrip(x_orig, kbps=kbps)
        else : 
            x_filtered_orig = x_orig
        logits = processor.classify(x_filtered_orig)
        
    current_logit = logits[0, y_true].item()
    
    # 원본의 성능을 초기 best 값으로 설정
    best_logit = current_logit
    best_rebuilt_audio = x_filtered_orig
    best_s_rebuilt = processor.stft(best_rebuilt_audio)
    best_num_removed = 0

    for num_to_remove in range(10, int(n_freqs * 0.5), 10):
        
        S_candidate = S_orig.clone()
         
        # 주파수 제거
        indices_to_remove = sorted_indices[:num_to_remove]
        S_candidate[:, indices_to_remove, :] = 0.0

        # 오디오 재구성 및 평가
        with torch.no_grad():
            x_rebuilt = processor.istft(S_candidate, length=T_wav)
            if kbps > 0 :
                x_filtered = processor.encodec_roundtrip(x_rebuilt, kbps=kbps)
            else :
                x_filtered = x_rebuilt
            logits = processor.classify(x_filtered)
        
        current_logit = logits[0, y_true].item()
        
        # 최고 점수 업데이트
        if current_logit > best_logit:
            best_logit = current_logit
            best_rebuilt_audio = x_filtered
            best_s_rebuilt = processor.stft(best_rebuilt_audio)
            best_num_removed = num_to_remove
            best_indices = indices_to_remove

    return best_rebuilt_audio, best_s_rebuilt, best_num_removed, best_indices

def load_wav_16k(path, target_len_min=400):
    wav, sr = torchaudio.load(path)
    
    if sr != SR: wav = AF.resample(wav, sr, SR)
    if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
    if wav.shape[-1] < target_len_min:
        pad = torch.zeros(1, target_len_min - wav.shape[-1])
        wav = torch.cat([wav, pad], dim=-1)
    return wav

# ------------------------------
# 3. 핵심 평가 함수
# ------------------------------
def evaluate_vip_reconstruction(test_dir, processor: AudioProcessor, kbps=1.5, keep_percent=50):
    fold = int(processor.clf_path.split("/")[-2][-1])
    print(f"Calculating Saliency with fold {fold} at Encodec rate {kbps}")
    paths = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
    if not paths:
        print(f"Warning: No .wav files found in '{test_dir}'")
        return

    output_dirs = {"filter_wins": "results/filter_wins", "baseline_wins": "results/baseline_wins", "both_fail": "results/both_fail"}     
    pruning_stats_records = []
    # total_freq_bins = N_FFT // 2 + 1
    # pruning_stats = []
    
    # Saliency 계산을 위한 함수
    def compute_saliency(wav, target_id):
        with torch.enable_grad():
            wav = wav.clone().detach().requires_grad_(True)
            processor.clf.requires_grad_(True)
            feat = None
            def hook(m, i, o):
                nonlocal feat; feat = o; feat.retain_grad()
            handle = processor.clf.beats.layer_norm.register_forward_hook(hook)
            
            logits = processor.clf(wav)
            loss = F.cross_entropy(logits, torch.tensor([target_id], device=DEVICE))
            loss.backward()
            
            handle.remove()
            processor.clf.requires_grad_(False) # 다시 grad 비활성화
            processor.clf.eval() # 평가 모드로 복귀

            if feat is None or feat.grad is None: return None
            saliency_2d = torch.relu(feat.detach() * feat.grad).squeeze(0)
            return saliency_2d / (saliency_2d.max() + 1e-9)

    total_files = 0

    
    progress_bar = tqdm(paths, desc="Saving Saliency-Masked Audio")
    for p in progress_bar :
        fn = os.path.basename(p)
        fn_base = os.path.splitext(fn)[0]
        if fn_base[0] != str(fold) : continue  # 현재 fold에 해당하는 파일만 처리
        y_true = processor.esc.label_from_filename(fn)
        if y_true is None: continue

        total_files += 1
        x_orig = load_wav_16k(p).to(DEVICE).unsqueeze(0)
        T_wav = x_orig.shape[-1]

        with torch.no_grad():
            S_orig = processor.stft(x_orig)
            # logits_orig = processor.classify(x_orig)
            # pred_orig = logits_orig.argmax(-1).item()


        saliency_2d_raw = compute_saliency(x_orig, y_true) # 중요: 실제 정답(y_true)을 기준으로 Saliency 계산
        if saliency_2d_raw is None: continue

        saliency_map_resized = F.interpolate(
            saliency_2d_raw.permute(1, 0).unsqueeze(0).unsqueeze(0),
            size=(S_orig.shape[1], S_orig.shape[2]),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        x_filter, S_filter, num_removed, removed_freqs = find_optimal_global_pruning(
        processor, x_orig, S_orig, T_wav, saliency_map_resized, y_true, kbps
        )
        
        # pruning_stats.append(num_removed)
        pruning_stats_records.append({
            "filename": fn,
            "label_id": y_true,
            "num_removed": num_removed,
            "total_freq_bins": S_orig.shape[1],
            "percent_removed": (num_removed / (S_orig.shape[1])) * 100,
            "removed_freqs": removed_freqs.cpu().tolist()
        })
        
        if x_filter is None:
            print(f"Skipping {fn} due to pruning failure.")
            continue
        
        torchaudio.save(os.path.join(test_dir+"_sal_filtered_no_encodec", f"{fn_base}.wav"), x_filter.cpu().squeeze(0), SR)
        
        
    # # --- Save pruning stats to CSV ---
    # if pruning_stats_records:
    #     csv_path = os.path.join(test_dir+"_sal_filtered_no_encodec", "pruning_stats.csv")
    #     pd.DataFrame(pruning_stats_records).to_csv(csv_path, index=False)
    #     print(f"Saved detailed pruning stats to {csv_path}")
    # print("--------------------------------")
    
    
    # --- Save pruning stats to JSON ---
    if pruning_stats_records:
        json_path = os.path.join(test_dir + "_sal_filtered_no_encodec", "pruning_stats.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(pruning_stats_records, f, indent=4, ensure_ascii=False)
        print(f"Saved detailed pruning stats to {json_path}")

    print("--------------------------------")


def main():
    ap = argparse.ArgumentParser(description="Evaluate the 'Saliency VIP Reconstruction' method for Encodec robustness.")
    ap.add_argument("--test_dir", default="/data/ESC-50-master/audio")
    ap.add_argument("--esc50_meta_path", default="/data/ESC-50-master/meta/esc50.csv")
    ap.add_argument("--beats_ckpt_path", default="/data/BEATs/BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt1.pt", help="BEATs 체크포인트 경로")
    ap.add_argument("--clf_path", default="/data/ckpts/ESC-50/baseline_16K/fold_5/latest.pt", help="학습된 분류기 헤드 체크포인트 경로")
    ap.add_argument("--kbps", type=float, default=1.5, help="Encodec bitrate, give 0 for no EnCodec usage")
    ap.add_argument("--keep_percent", type=int, default=50, help="각 프레임에서 보존할 상위 Saliency 주파수 비율(%)")

    args = ap.parse_args()
    
    os.makedirs(args.test_dir+"_sal_filtered/br_"+str(args.kbps), exist_ok=True)

    processor = AudioProcessor(
        cfg=args,
        esc50_csv=args.esc50_meta_path,
        beats_ckpt_path=args.beats_ckpt_path,
        clf_path=args.clf_path
    )

    evaluate_vip_reconstruction(
        args.test_dir, 
        processor, 
        kbps=args.kbps,
        keep_percent=args.keep_percentx
    )