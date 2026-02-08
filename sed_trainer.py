import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
import argparse
import numpy as np
import random
import pandas as pd
import os

from omegaconf import OmegaConf

from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict
import sed_eval

from data.dataset import StronglyAnnotatedSet, WeakSet
from data.ManyHotEncoder import *

from baselines.sound_event_detection.pipelines import BaseEventDetector, EnCodecEventDetector, FilterEventDetector

def to_clip_labels(labels, num_classes):
    assert labels.dim() == 3, f"expected 3D labels, got {labels.shape}"
    if labels.shape[1] == num_classes:  # [B, C, T]
        clip = labels.amax(dim=2)       # [B, C]
    else:                               # [B, T, C]
        clip = labels.transpose(1, 2).amax(dim=2)  # -> [B, C]
    return clip

class BeatsSEDTrainer:
    def __init__(self, cfg):
        
        # Fix seed
        
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.data_dir = cfg.data_dir
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self._init_models()
        
        if self.cfg.baseline.load_pretrained :
            self.load_pretrained(self.cfg.baseline.load_pretrained)
        
        # Initialize data
        self._init_data()
        
        # Initialize optimizer and scheduler
    
        
        if self.cfg.log.wandb :
            self.init_wandb()
        
        # Initialize loss functions
        if self.cfg.mode == "strong" : 
            raise NotImplementedError
            # self.loss_fn = nn.BCEWithLogitsLoss()
        if self.cfg.mode == "weak" : 
            self.loss_fn = nn.BCEWithLogitsLoss()
        
        self.epoch = 0
        

        # Metrics tracking
        self.best_strong_f1 = 0.0
        self.best_weak_f1 = 0.0
        self.train_losses = []
        self.val_metrics = []
        
        # Create checkpoint directory
        self.checkpoint_dir = cfg.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        
    def _init_models(self):
        """Initialize strong and weak models"""
        
        print("Initializing models...")
        
        # Strong label model
        
        if self.cfg.mode == 'strong' : 
            print(f"Strong model parameters: {sum(p.numel() for p in self.strong_model.parameters() if p.requires_grad):,}")
        
            raise NotImplementedError
            # self.model = BaseEventDetector(
            #     self.cfg,
            #     num_classes=len(self.cfg.data.classes),
            #     mode='strong'
            # ).to(self.device)
            
        
        # Weak label model (for evaluation)
        elif self.cfg.mode == 'weak' :
            
            if self.cfg.filters :
                self.model = FilterEventDetector(
                    self.cfg,
                    num_classes=len(self.cfg.data.classes),
                    mode='weak'
                ).to(self.device)
                
            elif self.cfg.encodec :
                self.model = EnCodecEventDetector(
                    self.cfg,
                    num_classes=len(self.cfg.data.classes),
                    mode='weak'
                ).to(self.device)


            else : 
                self.model = BaseEventDetector(
                    self.cfg,
                    num_classes=len(self.cfg.data.classes),
                    mode='weak'
                ).to(self.device)
        
            print(f"model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
    def _init_data(self):
        """Initialize datasets and dataloaders"""

        print("Loading datasets...")
        
        # Load metadata
        train_tsv = pd.read_csv(
            self.cfg.annotation_dir,
            sep="\t"
        )
        val_tsv = pd.read_csv(
            self.cfg.annotation_dir.replace("train", "eval"),
            sep="\t"
        )
        
        # Initialize encoder
        self.encoder = ManyHotEncoder(
            labels=self.cfg.data.classes,
            n_frames = self.cfg.data.n_frames
            # audio_len=self.cfg.data.audio_max_len,
            # frame_len=self.cfg.data.feats_frame_len,
            # frame_hop=self.cfg.data.feats_frame_hop,
            # net_pooling=self.cfg.data.net_pooling,
            # fs=self.cfg.data.fs
        )
        
        # Create datasets
        self.train_dataset = StronglyAnnotatedSet(
            #audio_folder= self.data_dir, #Path(self.data_dir) / "strong_label_real",
            audio_folder= Path(self.data_dir) / "strong_label_real",
            tsv_entries=train_tsv,
            encoder=self.encoder,
            pad_to=self.cfg.data.audio_max_len,
            fs=self.cfg.data.fs,
            return_filename=True
        )
        
        # self.train_dataset = WeakSet(
        #     audio_folder=Path(self.data_dir) / "audio" / "train" / "strong_label_real",
        #     tsv_entries=train_tsv,
        #     encoder=self.encoder,
        #     pad_to=self.cfg.data.audio_max_len,
        #     fs=self.cfg.data.fs,
        #     return_filename=True
        # )
        
        self.val_dataset = StronglyAnnotatedSet(
            # audio_folder= self.data_dir, #Path(self.data_dir.replace("train", "eval")) / "strong_label_real",
            audio_folder= Path(self.data_dir.replace("train", "eval")) / "strong_label_real",
            tsv_entries=val_tsv,
            encoder=self.encoder,
            pad_to=self.cfg.data.audio_max_len,
            fs=self.cfg.data.fs,
            return_filename=True,
            test=True
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay
        )
        
        self.scheduler = None
        
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer,
        #     T_max=self.cfg.train.epochs,
        #     eta_min=self.cfg.train.min_lr
        # )
        
    def init_wandb(self) :
        wandb.init(
            project=self.cfg.log.process_name,         # your project name
            name=f"bitrate_{self.cfg.baseline.bitrate}",      # run name
            config=OmegaConf.to_container(self.cfg, resolve=True)  # log all config values
        )
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch = self.epoch
        epoch_loss = 0.0
        epoch_acc = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.train.epochs}")
        
        for i, batch_data in enumerate(pbar):
            # Unpack batch
            audio = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            
            # Forward pass
            logits, _, _, _ = self.model(audio)
            
            labels_clip = to_clip_labels(labels, num_classes=10).to(logits.dtype).to(logits.device)

            # Compute loss
            loss = self.loss_fn(logits, labels_clip)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            # if self.cfg.train.get('grad_clip', None):
            #     torch.nn.utils.clip_grad_norm_(
            #         self.model.parameters(),
            #         self.cfg.train.grad_clip
            #     )
            
            self.optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            if self.cfg.log.wandb :
                wandb.log({"train/loss": loss.item() / (i+1), "steps": i})
              
                    # ===== Compute batch accuracy =====
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()  # [B, C]
                correct = (preds == labels_clip).float().mean()  # mean over all elements
                batch_acc = correct.item()
        
            epoch_acc += batch_acc

            # Update tqdm bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{batch_acc*100:.2f}%"
            })
            
            

        
        
        avg_loss = epoch_loss / len(self.train_loader)
        avg_acc = epoch_acc / len(self.train_loader)
        self.train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Train Acc: {avg_acc*100:.2f}%")
        self.epoch += 1
        
        if self.cfg.log.wandb :
            wandb.log({
                "train/loss": avg_loss,
                "steps": i,
                "train/accuracy": avg_acc})
                
        
        
        return avg_loss
    
    def evaluate(self):
        """Evaluate on validation set with both strong and weak metrics"""
        epoch = self.epoch
        self.model.eval()

        
        # Storage for predictions and ground truth
        # all_strong_preds = []
        # all_strong_labels = []
        all_weak_preds = []
        all_weak_labels = []
        all_filenames = []
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Evaluating"):
                # Unpack batch
                audio = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)
                filenames = batch_data[3] if len(batch_data) > 3 else None
                
                
                
                # Strong predictions
                strong_logits, _, _, _ = self.model(audio)
                strong_probs = torch.sigmoid(strong_logits)
                
                weak_labels = to_clip_labels(labels, num_classes=10).to(strong_logits.dtype).to(strong_logits.device)
                weak_probs = (torch.sigmoid(strong_logits) > 0.5).float()

                # print(weak_labels.shape)
                # print(weak_probs.shape)
                
                # Weak labels (clip-level)
                # weak_labels = (labels.sum(dim=1) > 0).float()
                
                # Weak predictions (max pooling over time)
                # weak_probs = strong_probs.max(dim=1)[0]
                
                # Store predictions
                # all_strong_preds.append(strong_probs.cpu())
                # labels_clip.append(labels_clip.cpu())
                all_weak_preds.append(weak_probs.cpu())
                all_weak_labels.append(weak_labels.cpu())
                
                if filenames:
                    all_filenames.extend(filenames)
        
        # Concatenate all predictions
        # all_strong_preds = torch.cat(all_strong_preds, dim=0)
        # all_strong_labels = torch.cat(all_strong_labels, dim=0)
        all_weak_preds = torch.cat(all_weak_preds, dim=0)
        all_weak_labels = torch.cat(all_weak_labels, dim=0)
        
        # print(all_weak_preds)
        # print(all_weak_labels[0]) # classes
        
        # Compute strong label metrics
        # strong_metrics = self._compute_strong_metrics(
        #     all_strong_preds,
        #     all_strong_labels,
        #     threshold=0.5
        # )
        strong_metrics=0
        
        # Compute weak label metrics
        weak_metrics = self._compute_weak_metrics(
            all_weak_preds,
            all_weak_labels,
            threshold=0.5
        )
        
        # print(all_weak_preds)
        # print(all_weak_labels)
        
        # Log metrics
        print(f"\nEpoch {epoch+1} Validation Results:")
        # print(f"Strong - F1: {strong_metrics['f1']:.4f}, Precision: {strong_metrics['precision']:.4f}, Recall: {strong_metrics['recall']:.4f}")
        print(f"""Weak   -
              Acc: {weak_metrics['accuracy']:.4f},
              F1: {weak_metrics['f1']:.4f}, 
              Precision: {weak_metrics['precision']:.4f}, 
              Recall: {weak_metrics['recall']:.4f}""")
        
        
        # Store metrics
        self.val_metrics.append({
            'epoch': epoch + 1,
            'strong': strong_metrics,
            'weak': weak_metrics
        })
        
        metrics_json_path = "validation_metrics.json"
        with open(metrics_json_path, 'w') as f:
            json.dump(self.val_metrics, f, indent=2)
        
        if self.cfg.log.wandb :
            wandb.log({"val/accuracy": weak_metrics['accuracy'], 
                    "val/f1" : weak_metrics['f1'],
                    "val/precision" : weak_metrics['precision'],
                    "val/recall" : weak_metrics['recall']})
        
        return strong_metrics, weak_metrics
    
    def _compute_strong_metrics(self, preds, labels, threshold=0.5):
        """Compute frame-level metrics for strong labels"""
        # Binarize predictions
        preds_binary = (preds > threshold).float()
        
        # Compute metrics per class then average
        tp = (preds_binary * labels).sum(dim=(0, 1))
        fp = (preds_binary * (1 - labels)).sum(dim=(0, 1))
        fn = ((1 - preds_binary) * labels).sum(dim=(0, 1))
        
        # Micro-averaged metrics
        precision = tp.sum() / (tp.sum() + fp.sum() + 1e-10)
        recall = tp.sum() / (tp.sum() + fn.sum() + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item()
        }
    
    def _compute_weak_metrics(self, preds, labels, threshold=0.5):
        """Compute clip-level metrics for weak labels"""
        # Binarize predictions
        preds_binary = (preds > threshold).float()
        
        #sanity check
        correct = (preds == preds_binary).float().mean()  # mean over all elements
        batch_acc = correct.item()
        # print(batch_acc)

        # Compute metrics
        tp = (preds_binary * labels).sum()
        fp = (preds_binary * (1 - labels)).sum()
        fn = ((1 - preds_binary) * labels).sum()
        tn = ((1 - preds_binary) * (1 - labels)).sum()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        # Micro accuracy = (TP + TN) / (TP + TN + FP + FN)
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-10)

        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'accuracy': acc.item()
        }
    
    def save_checkpoint(self, epoch, strong_metrics, weak_metrics, is_best_strong=False, is_best_weak=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
            'strong_model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            # 'strong_metrics': strong_metrics,
            'weak_metrics': weak_metrics,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'config': self.cfg
        }
        
        # Save latest checkpoint
        # latest_path = os.path.join(self.checkpoint_dir,f"br_{self.cfg.baseline.bitrate}", "latest.pt")
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoints
        # if is_best_strong:
        #     best_strong_path = self.checkpoint_dir / "best_strong.pt"
        #     torch.save(checkpoint, best_strong_path)
        #     print(f"Saved best strong model with F1: {strong_metrics['f1']:.4f}")
        
        if is_best_weak:
            # best_weak_path = os.path.join(self.checkpoint_dir, f"br_{self.cfg.baseline.bitrate}", "best_weak.pt")
            best_weak_path = os.path.join(self.checkpoint_dir, "best_weak.pt")
            torch.save(checkpoint, best_weak_path)
            print(f"Saved best weak model with F1: {weak_metrics['f1']:.4f}")
    
    def train(self):
        self._init_optimizer()
        """Main train loop"""
        print(f"\nStarting train on {self.device}")
        print(f"Total epochs: {self.cfg.train.epochs}")
        print(f"Batch size: {self.cfg.train.batch_size}")
        print(f"Learning rate: {self.cfg.train.learning_rate}\n")
        
        for epoch in range(self.cfg.train.epochs):
            # Train
            train_loss = self.train_epoch()
            print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")
            
            # Evaluate
            _, weak_metrics = self.evaluate()
            
            # Update learning rate
            if self.scheduler is not None: 
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Learning rate: {current_lr:.6f}")
            
            # Check if best model
            # is_best_strong = strong_metrics['f1'] > self.best_strong_f1
            is_best_weak = weak_metrics['f1'] > self.best_weak_f1
            
            # if is_best_strong:
                # self.best_strong_f1 = strong_metrics['f1']
            if is_best_weak:
                self.best_weak_f1 = weak_metrics['f1']
            
            # Save checkpoint
            # self.save_checkpoint(
            #     epoch,
            #     strong_metrics,
            #     weak_metrics,
            #     is_best_strong,
            #     is_best_weak
            # )
            self.save_checkpoint(
                epoch,
                None,
                weak_metrics,
                None,
                is_best_weak
            )
            
            # Save metrics to JSON
            metrics_file = os.path.join(self.checkpoint_dir, "metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump({
                    'train_losses': self.train_losses,
                    'val_metrics': self.val_metrics,
                    'best_strong_f1': None, #self.best_strong_f1,
                    'best_weak_f1': self.best_weak_f1
                }, f, indent=2)
            
            # print(f"Best Strong F1: {self.best_strong_f1:.4f} | Best Weak F1: {self.best_weak_f1:.4f}\n")
            print(f"Best Weak F1: {self.best_weak_f1:.4f}\n")
        
        print("Training completed!")
        # print(f"Best Strong F1: {self.best_strong_f1:.4f}")
        print(f"Best Weak F1: {self.best_weak_f1:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.strong_model.load_state_dict(checkpoint['strong_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Strong F1: {checkpoint['strong_metrics']['f1']:.4f}")
        print(f"Weak F1: {checkpoint['weak_metrics']['f1']:.4f}")
        
        return checkpoint['epoch']
    
    def load_pretrained(self, path):
        print(f"Loading pretrained model from {path}")
        
        if path.endswith('.pt') :
            checkpoint = torch.load(path, map_location=self.device)
        else :
            checkpoint = torch.load(os.path.join(path, "latest.pt"), map_location=self.device)
        
        self.model.load_state_dict(checkpoint['strong_model_state_dict'], strict=False)
    


# Example usage and configuration
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default="configs/detection_AudioSet.yaml")
    parser.add_argument("--bitrate", type=float, default=None)
    parser.add_argument("--load_ckpt", type=str, default=None)
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.configs)
    
    
    if args.bitrate is not None :
        cfg.baseline.bitrate = args.bitrate
        
    # if args.load_ckpt is not None :
    #     cfg.baseline.load_pretrained = args.load_ckpt
    
    trainer = BeatsSEDTrainer(cfg)
    
    # Train
    trainer.evaluate()
    trainer.train()