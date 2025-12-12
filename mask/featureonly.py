import os, glob, argparse, logging, math
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from baselines.beats.BEATs import BEATs, BEATsConfig
from encodec import EncodecModel
from sklearn.metrics import f1_score

# ------------------------------
# Globals
# ------------------------------
SR = 16000
N_FFT = 1024
HOP = 256
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

def load_wav_16k(path, target_len_min=400):
    wav, sr = torchaudio.load(path)
    if sr != SR: wav = AF.resample(wav, sr, SR)
    if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
    if wav.shape[-1] < target_len_min:
        pad = torch.zeros(1, target_len_min - wav.shape[-1])
        wav = torch.cat([wav, pad], dim=-1)
    return wav

def list_wavs(folder):
    return sorted(glob.glob(os.path.join(folder, "*.wav")))

# ------------------------------
# Models & Processor
# ------------------------------
class BEATsClassifier(nn.Module):
    def __init__(self, beats_ckpt_path, num_classes, freeze_beats=True):
        super().__init__()
        ckpt = torch.load(beats_ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]; cfg["finetuned_model"] = False
        self.beats = BEATs(BEATsConfig(cfg))
        self.beats.load_state_dict(ckpt["model"], strict=False)
        if freeze_beats:
            for p in self.beats.parameters(): p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(self.beats.cfg.encoder_embed_dim, 256),
            nn.Dropout(0.2), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, padding_mask=None):
        x = x.squeeze(1)
        feats = self.beats.extract_features(x, padding_mask=padding_mask)[0]
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
        except Exception: return None

class AudioProcessor:
    def __init__(self, esc50_csv, beats_ckpt_path, clf_path=None, sr=SR, n_fft=N_FFT, hop=HOP):
        self.device = DEVICE
        self.sr = sr
        self.esc = ESC50Info(esc50_csv)
        self.num_classes = len(self.esc.label2id)
        self.clf = BEATsClassifier(beats_ckpt_path, self.num_classes, freeze_beats=True).to(self.device)
        self.clf.eval()
        if clf_path and os.path.isfile(clf_path):
            sd = torch.load(clf_path, map_location=self.device, weights_only=False)
            state = sd.get('model_state_dict', sd)
            head = {k[len('classifier.'):]: v for k, v in state.items() if k.startswith('classifier.')}
            self.clf.classifier.load_state_dict(head, strict=False)
            logging.info(f"Loaded classifier head: {clf_path}")
        else: logging.warning("Classifier head not loaded.")

        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop, power=None, return_complex=True).to(self.device)
        self.ispec = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, hop_length=hop).to(self.device)
        self.encodec = EncodecModel.encodec_model_24khz().to(self.device).eval()

    def stft(self, wav): return self.spec(wav.squeeze(1))
    def istft(self, S, length): return self.ispec(S, length=length).unsqueeze(1)
    
    @torch.no_grad()
    def encodec_roundtrip(self, wav16, kbps):
        self.encodec.set_target_bandwidth(kbps)
        B, _, T_wav = wav16.shape
        if T_wav == 0: return torch.zeros_like(wav16)
        wav24 = AF.resample(wav16, SR, 24000)
        enc = self.encodec.encode(wav24)
        y24 = self.encodec.decode(enc)
        y16 = AF.resample(y24, 24000, SR)
        if y16.shape[-1] != T_wav:
            if y16.shape[-1] > T_wav: y16 = y16[..., :T_wav]
            else:
                pad = torch.zeros(B, 1, T_wav - y16.shape[-1], device=y16.device)
                y16 = torch.cat([y16, pad], dim=-1)
        return y16
    
    def _make_padding_mask(self, lengths, max_len):
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idx >= lengths.unsqueeze(1)

    @torch.no_grad()
    def classify(self, wav, lengths=None):
        if lengths is None: return self.clf(wav, padding_mask=None)
        padding_mask = self._make_padding_mask(lengths, wav.shape[-1])
        return self.clf(wav, padding_mask=padding_mask)

class AudioDataset(Dataset):
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        wav = load_wav_16k(path).squeeze(0)
        label = self.processor.esc.label_from_filename(os.path.basename(path))
        return wav, label, path

def collate_fn(batch):
    wavs, labels, paths = zip(*batch)
    lengths = torch.tensor([len(w) for w in wavs], dtype=torch.long)
    wavs_padded = pad_sequence(wavs, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.long)
    return wavs_padded.unsqueeze(1), labels, lengths, list(paths)

# ------------------------------
# 1. FeatureOnlyFilter (Fast)
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
        handle = self.processor.clf.beats.layer_norm.register_forward_hook(hook)
        with torch.no_grad(): _ = self.processor.clf(wav_batch, padding_mask=None)
        handle.remove()

        if feat is None: return None, None

        B, F_target, T_target = S_orig.shape
        token_score = feat.abs().mean(dim=-1)
        maps_list = []
        for i in range(B):
            ts = token_score[i]
            F_prime = _infer_grid_Fprime(ts.numel(), self.processor.clf.beats.cfg, F_target)
            N_prime = ts.numel() // F_prime
            grid = ts[:F_prime*N_prime].view(F_prime, N_prime)
            sal = F.interpolate(
                grid.unsqueeze(0).unsqueeze(0), size=(F_target, T_target),
                mode='bilinear', align_corners=False
            ).squeeze()
            if sal.max() > 0: sal = sal / sal.max()
            maps_list.append(sal)
        
        sal_batch = torch.stack(maps_list).to(self.device)
        return S_orig, sal_batch

# ------------------------------
# 2. Legacy Saliency (Gradient)
# ------------------------------
def compute_single_legacy_saliency(wav, target_id, processor, S_ref):
    # Original Gradient-based method
    with torch.enable_grad():
        wav = wav.clone().detach().requires_grad_(True)
        feat = None
        def hook(m, i, o):
            nonlocal feat
            feat = o
            feat.retain_grad()
        handle = processor.clf.beats.layer_norm.register_forward_hook(hook)
        logits = processor.clf(wav)
        loss = F.cross_entropy(logits, torch.tensor([target_id], device=wav.device))
        processor.clf.zero_grad(set_to_none=True)
        loss.backward()
        handle.remove()

        if feat is None or feat.grad is None: return None
        g = torch.relu((- feat.detach() * feat.grad).squeeze(0))
        token_score = g.mean(dim=1)
        F_target, T_target = S_ref.shape[1], S_ref.shape[2]
        F_prime = _infer_grid_Fprime(token_score.numel(), processor.clf.beats.cfg, F_target)
        N_prime = token_score.numel() // F_prime
        token_score = token_score[:F_prime*N_prime]
        grid = token_score.view(F_prime, N_prime)
        sal2d = F.interpolate(grid.unsqueeze(0).unsqueeze(0), size=(F_target, T_target), mode='bilinear', align_corners=False).squeeze()
        m = sal2d.max()
        if torch.isfinite(m) and m > 0: sal2d = sal2d / m
        return sal2d

# ------------------------------
# 3. Common Optimization Logic
# ------------------------------
def optimize_pruning_by_step(processor, S_orig, lengths, sal_map, logits_base, y_true, kbps, step):
    """
    주어진 Saliency Map(sal_map)을 사용하여 0~50%를 step단위로 자르며 최적의 오디오를 찾음.
    Fast와 Legacy가 이 함수를 공유함.
    """
    B, F, T = sal_map.shape
    freq_scores = sal_map.mean(dim=2) # (B, F)
    max_remove = int(F * 0.5)
    
    best_logits = torch.full((B,), -1e9, device=DEVICE)
    best_preds = logits_base.argmax(1).clone() # 초기값: Baseline 예측
    
    # 0은 포함해도 되고 안해도 됨(Baseline과 같음). 여기선 포함.
    pruning_counts = range(0, max_remove + 1, step)

    for k in pruning_counts:
        # (A) Masking
        mask = torch.ones((B, F, 1), device=DEVICE)
        if k > 0:
            sorted_idx = torch.argsort(freq_scores, dim=1, descending=False)
            low_k_idx = sorted_idx[:, :k]
            mask.scatter_(1, low_k_idx.unsqueeze(-1), 0.0)
        
        # (B) Reconstruct
        S_filt = S_orig * mask
        x_recon_list = []
        for i in range(B):
            wav = processor.istft(S_filt[i:i+1], length=lengths[i])
            x_recon_list.append(wav.squeeze(0).squeeze(0))
        x_recon = pad_sequence(x_recon_list, batch_first=True).unsqueeze(1)

        # (C) Classify
        with torch.no_grad():
            x_cand = processor.encodec_roundtrip(x_recon, kbps=kbps)
            logits_cand = processor.classify(x_cand, lengths)
            
            # (D) Oracle Update
            true_logits = logits_cand.gather(1, y_true.view(-1,1)).squeeze(1)
            improved = true_logits > best_logits
            if improved.any():
                best_logits[improved] = true_logits[improved]
                best_preds[improved] = logits_cand.argmax(1)[improved]
                
    return best_preds

# ------------------------------
# ★ 3-Way Comparison Function
# ------------------------------
def evaluate_compare_three_methods(audio_folder, test_fold, processor, kbps, 
                                   batch_size=4, step=10):
    paths = list_wavs(audio_folder)
    paths = [p for p in paths if os.path.basename(p).startswith(f"{test_fold}-")]
    if not paths: return None

    dataset = AudioDataset(paths, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)

    fast_filter = FeatureOnlyFilter(processor)
    
    total = 0
    base_correct = 0
    fast_correct = 0
    legacy_correct = 0
    
    logging.info(f"[Comparison] F{test_fold} @ {kbps}kbps | Step: {step} bins")
    logging.info("Method 1: Baseline (Encodec Only)")
    logging.info("Method 2: Fast Optimal (FeatureOnly Saliency)")
    logging.info("Method 3: Legacy Optimal (Gradient Saliency)")

    for x_orig, y_true, lengths, _ in tqdm(dataloader, desc="3-Way Eval"):
        x_orig = x_orig.to(DEVICE)
        y_true = y_true.to(DEVICE)
        lengths = lengths.to(DEVICE)
        B = x_orig.shape[0]
        total += B

        # --- 1. Baseline ---
        with torch.no_grad():
            x_base = processor.encodec_roundtrip(x_orig, kbps=kbps)
            logits_base = processor.classify(x_base, lengths)
            base_correct += (logits_base.argmax(1) == y_true).sum().item()

        # --- 2. Fast Method ---
        S_orig, sal_fast = fast_filter.get_batch_saliency(x_orig)
        if sal_fast is not None:
            pred_fast = optimize_pruning_by_step(
                processor, S_orig, lengths, sal_fast, logits_base, y_true, kbps, step
            )
            fast_correct += (pred_fast == y_true).sum().item()
        else:
            fast_correct += (logits_base.argmax(1) == y_true).sum().item()

        # --- 3. Legacy Method (On-the-fly) ---
        # Note: This is slower because of gradient calculation per sample
        legacy_maps = []
        for i in range(B):
            # Gradient Saliency를 위해 requires_grad 켠 상태로 호출
            # 이미 S_orig[i]가 있으니 차원 맞춰서 전달
            S_ref_i = S_orig[i].unsqueeze(0) # (1,F,T)
            sal_l = compute_single_legacy_saliency(x_orig[i].unsqueeze(0), y_true[i], processor, S_ref_i)
            if sal_l is None:
                # 실패시 Fast Map이나 1로 대체
                sal_l = torch.ones_like(sal_fast[i])
            legacy_maps.append(sal_l)
        
        sal_legacy = pad_sequence(legacy_maps, batch_first=True).to(DEVICE)
        # 차원 맞춤
        if sal_legacy.shape[-1] != S_orig.shape[-1]:
             sal_legacy = F.interpolate(sal_legacy.unsqueeze(1), size=S_orig.shape[1:], mode='nearest').squeeze(1)

        pred_legacy = optimize_pruning_by_step(
            processor, S_orig, lengths, sal_legacy, logits_base, y_true, kbps, step
        )
        legacy_correct += (pred_legacy == y_true).sum().item()

    return {
        "baseline_acc": 100.0 * base_correct / total,
        "fast_acc": 100.0 * fast_correct / total,
        "legacy_acc": 100.0 * legacy_correct / total
    }

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_root", default="data/ESC-50-master/audio")
    ap.add_argument("--saliency_root", default="data/ESC-50-master/saliency_hook")
    ap.add_argument("--esc50_meta_path", default="data/ESC-50-master/meta/esc50.csv")
    ap.add_argument("--beats_ckpt_path", default="data/BEATs/BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt1.pt")
    ap.add_argument("--clf_root", default="data/ckpts/ESC-50/res_post_only/post_percept_rev_bitrate6")
    
    ap.add_argument("--folds", nargs="*", type=int, default=[5])
    ap.add_argument("--kbps_list", nargs="*", type=float, default=[1.5])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--step", type=int, default=10, help="Pruning step (bins)")
    
    # Modes
    ap.add_argument("--use_fast_filter", action="store_true", help="Run only Fast Optimal")
    ap.add_argument("--compare_all", action="store_true", help="Compare Baseline vs Fast vs Legacy")
    
    args = ap.parse_args()
    set_seed(42)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

    for fold in args.folds:
        clf_path = os.path.join(args.clf_root, f"fold_{fold}", "ep_10.pt")
        processor = AudioProcessor(
            esc50_csv=args.esc50_meta_path,
            beats_ckpt_path=args.beats_ckpt_path,
            clf_path=clf_path if os.path.exists(clf_path) else None,
        )

        # 1. 3-Way Comparison
        if args.compare_all:
            logging.info(f"⚔️ STARTING 3-WAY COMPARISON (F{fold})")
            for kbps in args.kbps_list:
                res = evaluate_compare_three_methods(
                    audio_folder=args.audio_root,
                    test_fold=fold,
                    processor=processor,
                    kbps=kbps,
                    batch_size=args.batch_size,
                    step=args.step
                )
                if res:
                    print(f"\n[{fold} Fold @ {kbps} kbps Result]")
                    print(f"  - Baseline Acc      : {res['baseline_acc']:.2f}%")
                    print(f"  - Fast Optimal Acc  : {res['fast_acc']:.2f}% (Gain: {res['fast_acc']-res['baseline_acc']:+.2f}%)")
                    print(f"  - Legacy Optimal Acc: {res['legacy_acc']:.2f}% (Gain: {res['legacy_acc']-res['baseline_acc']:+.2f}%)")
                    print("-" * 40)

        # 2. Fast Optimal Only (Old option)
        elif args.use_fast_filter:
            # (기존 FeatureOnly 코드 실행)
             pass # 여기서는 생략하고 위 compare_all 사용을 권장
        
        else:
            print("Please use --compare_all to see the comparison.")

if __name__ == "__main__":
    main()