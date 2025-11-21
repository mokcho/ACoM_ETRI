import os, glob, argparse, logging, math, random
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

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
# Globals & Seed
# ------------------------------
SR = 16000
N_FFT = 1024
HOP = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = False 
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------------------
# BEATs & Encodec
# ------------------------------

class BEATsClassifier(nn.Module):
    def __init__(self, beats_ckpt_path, num_classes, freeze_beats=True):
        super().__init__()
        if not os.path.exists(beats_ckpt_path):
            raise FileNotFoundError(f"BEATs Checkpoint not found: {beats_ckpt_path}")
            
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

    def forward(self, x, padding_mask=None):
        x = x.squeeze(1)
        feats = self.beats.extract_features(x, padding_mask=padding_mask)[0]
        return self.classifier(feats.mean(dim=1))

# ------------------------------
# ESC-50 helpers
# ------------------------------
class ESC50Info:
    def __init__(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Meta CSV not found: {csv_path}")
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
        except Exception:
            return None

# ------------------------------
# Processor
# ------------------------------
class AudioProcessor:
    def __init__(self, esc50_csv, beats_ckpt_path, clf_path=None,
                 sr=SR, n_fft=N_FFT, hop=HOP):
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
            logging.info(f"✅ Loaded classifier head: {clf_path}")
        else:
            logging.warning(f"⚠️ Classifier head not loaded (Path: {clf_path}); using random init.")

        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop, power=None, return_complex=True
        ).to(self.device)
        self.ispec = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft, hop_length=hop
        ).to(self.device)

        self.encodec = EncodecModel.encodec_model_24khz().to(self.device).eval()
        self.resample_16_24 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000).to(self.device)
        self.resample_24_16 = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000).to(self.device)

    def stft(self, wav): return self.spec(wav.squeeze(1))
    
    def istft(self, S, length): return self.ispec(S, length=length).unsqueeze(1)

    @torch.no_grad()
    def encodec_roundtrip(self, wav16, kbps):
        self.encodec.set_target_bandwidth(kbps)
        B, _, T_wav = wav16.shape
        if T_wav == 0: return torch.zeros_like(wav16)
        
        wav24 = self.resample_16_24(wav16)
        enc = self.encodec.encode(wav24)
        y24 = self.encodec.decode(enc)
        y16 = self.resample_24_16(y24)
        
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

# ------------------------------
# Data utils (Modified for filtering)
# ------------------------------
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
# Hook-based saliency
# ------------------------------
def _infer_grid_Fprime(T_prime, cfg, F_target):
    ps = getattr(cfg, "input_patch_size", 16)
    ps_f = int(ps[0]) if isinstance(ps, (tuple, list)) else int(ps)
    F_mel = int(getattr(cfg, "input_fdim", 128))
    Fp_guess = max(1, F_mel // ps_f)
    if T_prime % Fp_guess == 0: return Fp_guess
    divisors = [d for d in range(1, min(T_prime, 64)+1) if T_prime % d == 0]
    return min(divisors, key=lambda d: abs(d - Fp_guess))

def compute_saliency_hook(wav, target_id, processor: AudioProcessor, S_ref):
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
        
        sal2d = F.interpolate(
            grid.unsqueeze(0).unsqueeze(0), size=(F_target, T_target),
            mode='bilinear', align_corners=False
        ).squeeze()
        
        m = sal2d.max()
        if torch.isfinite(m) and m > 0: sal2d = sal2d / m
        return sal2d

def precompute_saliency_with_hook(audio_folder, saliency_folder, processor: AudioProcessor, target_fold=None):
    os.makedirs(saliency_folder, exist_ok=True)
    
    # [수정] 모든 wav를 가져온 뒤, target_fold에 해당하는 것만 필터링
    all_paths = list_wavs(audio_folder)
    if target_fold is not None:
        paths = [p for p in all_paths if os.path.basename(p).startswith(f"{target_fold}-")]
        logging.info(f"[Saliency] Processing Fold {target_fold} files only ({len(paths)} files)")
    else:
        paths = all_paths
        logging.info(f"[Saliency] Processing ALL files ({len(paths)} files)")

    if not paths:
        logging.warning(f"[Saliency] No files found for fold {target_fold} in {audio_folder}")
        return

    for path in tqdm(paths, desc=f"Saliency(Hook) Fold{target_fold}"):
        fn = os.path.basename(path)
        out_pt = os.path.join(saliency_folder, fn.replace('.wav', '.pt'))
        if os.path.exists(out_pt): continue

        y_true = processor.esc.label_from_filename(fn)
        if y_true is None: continue

        wav = load_wav_16k(path).to(DEVICE).unsqueeze(0)
        with torch.no_grad(): S_ref = processor.stft(wav)
        sal = compute_saliency_hook(wav, y_true, processor, S_ref)
        if sal is None:
            logging.warning(f"[Fail] {fn}")
            continue
        torch.save(sal.cpu(), out_pt)

# ------------------------------
# Pruning (batched) & evaluation
# ------------------------------
def find_optimal_global_pruning_batched(
    processor: AudioProcessor, S_b, lengths_b, sal_map_b, y_true_b, kbps: float, prune_axis: str, step: int
):
    B, F, TT = S_b.shape
    if prune_axis == "freq":
        global_scores = sal_map_b.mean(dim=2)
        sorted_idx = torch.argsort(global_scores, dim=1, descending=False)
        axis_len = F
    else:
        global_scores = sal_map_b.mean(dim=1)
        sorted_idx = torch.argsort(global_scores, dim=1, descending=False)
        axis_len = TT

    max_remove = int(axis_len * 0.5)
    pruning_steps = torch.arange(0, max_remove + 1, step, device=S_b.device)
    K = len(pruning_steps)

    S_expanded = S_b.repeat_interleave(K, dim=0)
    
    mask_list = []
    if prune_axis == "freq": base_mask = torch.ones(B, F, 1, dtype=torch.bool, device=S_b.device)
    else: base_mask = torch.ones(B, 1, TT, dtype=torch.bool, device=S_b.device)

    for k in pruning_steps:
        curr_mask = base_mask.clone()
        if k > 0:
            lowk = sorted_idx[:, :k]
            if prune_axis == "freq": curr_mask.scatter_(1, lowk.unsqueeze(-1), False)
            else: curr_mask.scatter_(2, lowk.unsqueeze(1), False)
        mask_list.append(curr_mask)
    
    total_mask = torch.stack(mask_list, dim=1).view(B * K, *base_mask.shape[1:])
    S_cand_all = S_expanded * total_mask

    lengths_expanded = lengths_b.repeat_interleave(K)
    x_rebuilt_list = []
    for i in range(B * K):
        xb = processor.istft(S_cand_all[i:i+1], length=lengths_expanded[i].item())
        x_rebuilt_list.append(xb)
    x_rebuilt_all = torch.cat(x_rebuilt_list, dim=0)

    x_filtered_all = processor.encodec_roundtrip(x_rebuilt_all, kbps=kbps)
    logits_all = processor.classify(x_filtered_all, lengths_expanded)

    logits_reshaped = logits_all.view(B, K, -1)
    y_true_expanded = y_true_b.view(B, 1, 1).expand(-1, K, -1)
    true_logits = logits_reshaped.gather(2, y_true_expanded).squeeze(2)
    best_logit_vals, best_k_indices = true_logits.max(dim=1)

    best_audio_list = []
    best_removed_counts = []
    x_filtered_reshaped = x_filtered_all.view(B, K, 1, -1)

    for b in range(B):
        idx = best_k_indices[b].item()
        best_audio_list.append(x_filtered_reshaped[b, idx])
        best_removed_counts.append(pruning_steps[idx])

    best_audio = torch.stack(best_audio_list)
    best_removed = torch.tensor(best_removed_counts, device=S_b.device)

    return best_audio, best_removed, axis_len, sorted_idx


def evaluate_vip_reconstruction_axis(audio_folder, saliency_folder, test_fold, processor: AudioProcessor,
                                     kbps=1.5, batch_size=16, prune_axis="freq",
                                     step=20, max_archives_per_case=2000, out_dir_tag="freq"):
    # [수정] 전체 오디오 폴더에서 test_fold에 해당하는 파일만 필터링
    all_paths = list_wavs(audio_folder)
    paths = [p for p in all_paths if os.path.basename(p).startswith(f"{test_fold}-")]

    if not paths:
        logging.warning(f"No .wav starting with '{test_fold}-' in {audio_folder}")
        return None
    
    logging.info(f"[Eval] Fold {test_fold} selected {len(paths)} files from {audio_folder}")

    dataset = AudioDataset(paths, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)

    out_dirs = { "filter_wins": f"results/filter_wins_{out_dir_tag}/fold_{test_fold}",
                 "baseline_wins": f"results/baseline_wins_{out_dir_tag}/fold_{test_fold}",
                 "both_fail": f"results/both_fail_{out_dir_tag}/fold_{test_fold}" }
    for d in out_dirs.values(): os.makedirs(d, exist_ok=True)
    save_counts = defaultdict(int)

    total_files = baseline_correct = filter_correct = 0
    pruning_stats_pct = []
    all_true_labels = []
    all_filter_preds = []
    removal_order_data = []

    for x_orig_b, y_true_b, lengths_b, paths_b in tqdm(dataloader, desc=f"VIPs {prune_axis} F{test_fold} @{kbps}kbps"):
        x_orig_b = x_orig_b.to(DEVICE)
        y_true_b = y_true_b.to(DEVICE)
        lengths_b = lengths_b.to(DEVICE)
        B = x_orig_b.shape[0]
        total_files += B

        with torch.no_grad():
            S_orig_b = processor.stft(x_orig_b)
            x_baseline_b = processor.encodec_roundtrip(x_orig_b, kbps=kbps)
            logits_baseline_b = processor.classify(x_baseline_b, lengths_b)
            pred_baseline_b = logits_baseline_b.argmax(-1)
            baseline_correct += (pred_baseline_b == y_true_b).sum().item()

        saliency_maps = []
        for path in paths_b:
            fn = os.path.basename(path)
            sal_path = os.path.join(saliency_folder, fn.replace('.wav', '.pt'))
            if not os.path.exists(sal_path):
                # Saliency Map이 없으면 임시로 0 채움 (에러 방지)
                # raise FileNotFoundError(f"Missing saliency: {sal_path}")
                logging.warning(f"Missing saliency: {sal_path}, utilizing ones")
                saliency_maps.append(torch.ones(S_orig_b.shape[1], lengths_b[0], device=DEVICE)) # Dummy
            else:
                saliency_maps.append(torch.load(sal_path, map_location=DEVICE))
                
        sal_map_b = pad_sequence(saliency_maps, batch_first=True, padding_value=0.0).to(DEVICE)
        
        # Saliency 크기가 STFT 크기와 약간 다를 경우 보정 (Interpolation 이슈 방지)
        if sal_map_b.shape[-1] != S_orig_b.shape[-1]:
            sal_map_b = F.interpolate(sal_map_b.unsqueeze(1), size=(S_orig_b.shape[1], S_orig_b.shape[2]), mode='nearest').squeeze(1)

        x_filter_b, num_removed_b, axis_len, sorted_idx_b = find_optimal_global_pruning_batched(
            processor, S_orig_b, lengths_b, sal_map_b, y_true_b, kbps,
            prune_axis=prune_axis, step=step
        )

        if axis_len > 0: pruning_percentages_b = (num_removed_b.float() / axis_len) * 100
        else: pruning_percentages_b = torch.zeros_like(num_removed_b, dtype=torch.float)
        pruning_stats_pct.extend(pruning_percentages_b.detach().cpu().tolist())

        with torch.no_grad():
            logits_filter_b = processor.classify(x_filter_b, lengths_b)
            pred_filter_b = logits_filter_b.argmax(-1)
            filter_correct += (pred_filter_b == y_true_b).sum().item()
            all_true_labels.extend(y_true_b.cpu().tolist())
            all_filter_preds.extend(pred_filter_b.cpu().tolist())

        sorted_idx_list = sorted_idx_b.cpu().tolist()
        removed_counts = num_removed_b.cpu().tolist()
        
        for i in range(B):
            filename = os.path.basename(paths_b[i])
            full_removal_order = sorted_idx_list[i]
            best_k = removed_counts[i]
            actual_removed = full_removal_order[:best_k]
            removal_order_data.append({
                "filename": filename, "fold": test_fold, "kbps": kbps, "axis": prune_axis,
                "best_k_removed": best_k, "total_axis_len": axis_len,
                "full_removal_order_indices": str(full_removal_order),
                "actual_removed_indices": str(actual_removed)
            })

        # Archive cases (Optional)
        for i in range(B):
            base_ok = (pred_baseline_b[i] == y_true_b[i]).item()
            filt_ok = (pred_filter_b[i] == y_true_b[i]).item()
            case = None
            if (not base_ok) and filt_ok: case = "filter_wins"
            elif base_ok and (not filt_ok): case = "baseline_wins"
            if case and save_counts[case] < max_archives_per_case:
                fn_base = os.path.splitext(os.path.basename(paths_b[i]))[0]
                save_dir = out_dirs[case]
                # torchaudio.save(os.path.join(save_dir, f"{fn_base}_0_orig.wav"), x_orig_b[i].cpu(), SR)
                # torchaudio.save(os.path.join(save_dir, f"{fn_base}_1_baseline.wav"), x_baseline_b[i].cpu(), SR)
                # torchaudio.save(os.path.join(save_dir, f"{fn_base}_2_filter.wav"), x_filter_b[i].cpu(), SR)
                save_counts[case] += 1

    if removal_order_data:
        csv_name = f"results/removal_order_fold{test_fold}_{out_dir_tag}_{kbps}kbps.csv"
        try: pd.DataFrame(removal_order_data).to_csv(csv_name, index=False)
        except Exception as e: logging.error(f"Failed to save CSV: {e}")

    if total_files > 0:
        filter_f1 = f1_score(all_true_labels, all_filter_preds, average='macro', zero_division=0)
        return {
            "total_files": total_files,
            "avg_pruning": float(np.mean(pruning_stats_pct) if pruning_stats_pct else 0.0),
            "std_pruning": float(np.std(pruning_stats_pct) if pruning_stats_pct else 0.0),
            "baseline_acc": 100.0 * baseline_correct / total_files,
            "filter_acc": 100.0 * filter_correct / total_files,
            "filter_f1": filter_f1 * 100.0 
        }
    else:
        return None

# ------------------------------
# Orchestrator
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Hook-based saliency + VIP-Rebuild eval")
    # paths
    ap.add_argument("--audio_root", default="data/ESC-50-master/audio")
    ap.add_argument("--saliency_root", default="data/ESC-50-master/saliency_hook")
    ap.add_argument("--esc50_meta_path", default="data/ESC-50-master/meta/esc50.csv")
    ap.add_argument("--beats_ckpt_path", default="data/BEATs/BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt1.pt")
    ap.add_argument("--clf_root", default="data/ckpts/ESC-50/res_post_only/post_percept_rev_bitrate6")

    # experiment knobs
    ap.add_argument("--folds", nargs="*", type=int, default=[1])
    ap.add_argument("--kbps_list", nargs="*", type=float, default=[1.5])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--mode", choices=["saliency", "eval", "both"], default="both")
    ap.add_argument("--recompute_saliency", action="store_true")
    args = ap.parse_args()

    set_seed(42)
    os.makedirs("results", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

    summary_results = defaultdict(lambda: defaultdict(list))

    for fold in args.folds:
        # [수정] 오디오 폴더는 'foldX' 없이 그냥 root (모든 파일이 여기 있으므로)
        audio_folder = args.audio_root
        
        # [수정] Saliency와 Checkpoint는 여전히 Fold별로 구분되어 관리
        sal_folder = os.path.join(args.saliency_root, f"fold{fold}")
        clf_path = os.path.join(args.clf_root, f"fold_{fold}", "ep_10.pt")

        logging.info(f"\n========== PROCESS FOLD {fold} ==========")
        processor = AudioProcessor(
            esc50_csv=args.esc50_meta_path,
            beats_ckpt_path=args.beats_ckpt_path,
            clf_path=clf_path if os.path.exists(clf_path) else None,
        )

        # 1) Saliency
        if args.mode in ("saliency", "both"):
            if args.recompute_saliency and os.path.exists(sal_folder):
                for f in glob.glob(os.path.join(sal_folder, "*.pt")): os.remove(f)
            
            # [수정] 함수에 target_fold를 넘겨서 해당 fold 파일만 처리
            precompute_saliency_with_hook(audio_folder, sal_folder, processor, target_fold=fold)
            
            if args.mode == "saliency": continue

        # 2) Evaluate FREQ
        for kbps in args.kbps_list:
            logging.info(f"[Evaluate] FREQ Axis @ {kbps} kbps (Fold {fold})")
            res_f = evaluate_vip_reconstruction_axis(
                audio_folder=audio_folder, saliency_folder=sal_folder,
                test_fold=fold, processor=processor, kbps=kbps,
                batch_size=args.batch_size, prune_axis="freq", step=args.step, out_dir_tag="freq"
            )
            if res_f: summary_results["freq"][kbps].append(res_f)

        # 3) Evaluate TIME
        for kbps in args.kbps_list:
            logging.info(f"[Evaluate] TIME Axis @ {kbps} kbps (Fold {fold})")
            res_t = evaluate_vip_reconstruction_axis(
                audio_folder=audio_folder, saliency_folder=sal_folder,
                test_fold=fold, processor=processor, kbps=kbps,
                batch_size=args.batch_size, prune_axis="time", step=args.step, out_dir_tag="time"
            )
            if res_t: summary_results["time"][kbps].append(res_t)

    # 4) Final Summary
    for axis, kbps_results in summary_results.items():
        print(f"\n=== FINAL SUMMARY ({axis.upper()}) ===")
        for kbps, results in kbps_results.items():
            avg_filt_acc = np.mean([r['filter_acc'] for r in results])
            print(f"KBPS {kbps}: Avg Filter Acc: {avg_filt_acc:.2f}%")

if __name__ == "__main__":
    main()