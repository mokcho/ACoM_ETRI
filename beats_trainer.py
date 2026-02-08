import torch
import torch.nn.functional as F
import argparse
import os
import wandb
import numpy as np
import random

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize

from baselines.sound_classification.pipelines import *
from data import *


class BeatsTrainer:
    def __init__(self, cfg):
        
        # fix seed
        
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.data_dir = cfg.data_dir
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"-----------fold_{self.cfg.data.test_fold}------------")
        if 'AudioSet' in cfg.data_dir :
            self.train_dataset = AudioSetDataset(
                root_dir=self.data_dir,
                annotation_dir = os.path.join(cfg.annotation_dir, self.data_dir.split('/')[-1]+'.csv'),
                sample_rate=self.cfg.data.sr,
                max_duration=10
                )
            
            self.train_mode = "multi-label"
            
            self.criterion = nn.BCEWithLogitsLoss()
            self.consistency_criterion = nn.MSELoss()
            collate_fn = collate_fn_multi_label
        
        elif 'ESC-50' in cfg.data_dir :
            
            self.cfg.train_classifier = True
            
            self.train_dataset = ESCDataset(
                root_dir=self.data_dir,
                annotation_dir = self.cfg.annotation_dir,
                sample_rate=self.cfg.data.sr,
                test_fold = self.cfg.data.test_fold
                )
            
            self.eval_dataset = ESCDataset(
                root_dir = self.data_dir,
                annotation_dir = self.cfg.annotation_dir,
                sample_rate = self.cfg.data.sr,
                test_fold = self.cfg.data.test_fold,
                label2id = self.train_dataset.label2id,
                train = False
            )
            
            self.train_mode = "multi-class"
            collate_fn = collate_fn_multi_class
            
            self.criterion = nn.CrossEntropyLoss()
        
        else :
            print("DATASET NOT SUPPORTED")
            

        if self.cfg.filters :
            print("Filter training")
            self.pipeline = FilterClassifier(
                cfgs=cfg,
                label2id = self.train_dataset.label2id
                ).to(self.device)
        
        elif self.cfg.encodec: 
            print("EnCodec training")
            self.pipeline = EnCodecClassifier(
                cfgs=cfg,
                label2id= self.train_dataset.label2id
                ).to(self.device)
            
        else : 
            print("baseline training")
            self.pipeline = BaseClassifier(
                cfgs = cfg, 
                label2id = self.train_dataset.label2id
                ).to(self.device)
        
        self.augment_fn = WaveformAugmentations()
        
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size= self.cfg.train.batch_size,
            shuffle=True,
            num_workers= self.cfg.train.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
            )

        self.val_loader = DataLoader(
            dataset=self.eval_dataset,
            batch_size= self.cfg.train.batch_size,
            shuffle=False,
            num_workers= self.cfg.train.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        
        if self.cfg.mode == "train"  : 
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.pipeline.parameters()),
                                           lr=self.cfg.train.learning_rate)

        self.cur_epoch = 0
        
        if not self.cfg.train_classifier : 
            assert self.pipeline.baseline.cfg.predictor_class == len(self.train_dataset.label2id), f"{self.pipeline.baseline.cfg.predictor_class} does not match {len(self.train_dataset.label2id)}"

        if self.cfg.log.wandb :
            self.init_wandb()
        
        if self.cfg.baseline.load_pretrained :
            self.load_pretrained(self.cfg.baseline.load_pretrained)
            
        # self.check_params() #used for sanity check
    
    def check_params(self) :
        for name, param in self.pipeline.named_parameters() :
            if param.requires_grad :
                print(f"{name} is trainable")
            else :
                continue
                print(f"{name} is not trainable")

    
        
    def init_wandb(self) :
        wandb.init(
            project=self.cfg.log.process_name,         # your project name
            name=f"fold_{self.cfg.data.test_fold}_bitrate_{self.cfg.baseline.bitrate}",      # run name
            config=OmegaConf.to_container(self.cfg, resolve=True)  # log all config values
        )
        
    def train(self, epochs=10):
        total_epochs = self.cfg.train.epochs
        half_epochs = total_epochs // 2
        for epoch in range(total_epochs):
            # Switch trainable modules and optimizer at the halfway point
            if epoch == 0:
                # First half: train pre_filter only
                if isinstance(self.pipeline, FilterClassifier):
                    for name, param in self.pipeline.named_parameters():
                        if name.startswith('pre_filter'):
                            param.requires_grad = True
                        elif name.startswith('post_filter'):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.pipeline.parameters()),
                                                       lr=self.cfg.train.learning_rate)
            # elif epoch == half_epochs:
            #     # Second half: train post_filter only
            #     if isinstance(self.pipeline, FilterClassifier):
            #         for name, param in self.pipeline.named_parameters():
            #             if name.startswith('pre_filter'):
            #                 param.requires_grad = False
            #             elif name.startswith('post_filter'):
            #                 param.requires_grad = True
            #             else:
            #                 param.requires_grad = False
            #         self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.pipeline.parameters()),
            #                                            lr=self.cfg.train.learning_rate)
            self.pipeline.train()
            running_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.cur_epoch + 1}")
            
            all_logits = []
            all_targets = []
            for i, batch in enumerate(progress_bar):    
                audio, labels, _= batch
                audio = audio.to(self.device)
                
                if self.cfg.data.augmentation :
                    audio = self.augment_fn(audio)
                
                labels = labels.to(self.device)
                # Custom forward logic for FilterClassifier
                if isinstance(self.pipeline, FilterClassifier):
                    if epoch < half_epochs:
                        outs = self.pipeline(audio, use_post_filter=False)
                    else:
                        outs = self.pipeline(audio, use_post_filter=True)
                else:
                    outs = self.pipeline(audio)
                
                logits, x, features, orig_features = outs
                    
                if self.train_mode == "multi-label" :
                    labels = labels.float()
                    
                                
                    
                self.optimizer.zero_grad()

                # perception_loss = F.mse_loss(features, orig_features)  # x is the filtered waveform
                # loss = (1-self.cfg.train.percep_lambda) * self.criterion(logits, labels) + self.cfg.train.percep_lambda * perception_loss
                loss = self.criterion(logits, labels)
                
                loss.backward()
                self.optimizer.step()

                all_logits.append(logits.cpu())
                all_targets.append(labels.cpu())         
                
                running_loss += loss.item()
                progress_bar.set_postfix(loss=f"{running_loss / (i+1):.4f}")
                if self.cfg.log.wandb : 
                    wandb.log({"train/loss": running_loss / (i+1), 
                           #"perception_loss": perception_loss.item(), 
                           "steps": i})
              
            # print(f"Epoch {self.cur_epoch + 1} - Train Loss: {running_loss / len(self.train_loader):.4f}")
            
            all_logits = torch.cat(all_logits, dim=0)    # [N, C]
            all_targets = torch.cat(all_targets, dim=0)  # [N, C]
            
            # Compute mAP
            if self.train_mode == "multi-label" : 
                
                y_true = all_targets.numpy()
                y_pred = all_logits.numpy()
                mAP = average_precision_score(
                    y_true,
                    y_pred,
                    average='macro')
                
                print(f"mAP: {mAP:.4f}")
            
            elif self.train_mode == "multi-class" :
            
                y_true = all_targets.numpy()
                y_pred = all_logits.argmax(axis=1).numpy()
                

            
            # Accuracy
            
            acc = accuracy_score(
                y_true, y_pred)

            # F1 scores
            f1_macro = f1_score(
                y_true, y_pred, average='macro')  # good for class imbalance
            f1_micro = f1_score(
                y_true, y_pred, average='micro')  # good for overall precision/recall

            print(f"Accuracy: {acc:.4f}, F1 (macro): {f1_macro:.4f}, F1 (micro): {f1_micro:.4f}")
            
            
            if self.cfg.log.wandb : 
                wandb.log({
                    "train/f1_macro": f1_macro,
                    "train/f1_micro": f1_micro,
                    "epoch": self.cur_epoch
                })
                
            
            self.cur_epoch += 1
            self.save_checkpoint(name='latest')
            self.eval()
            
        self.save_checkpoint()

    def eval(self):
        total_epochs = self.cfg.train.epochs
        half_epochs = total_epochs // 2
        
        self.pipeline.eval()
        
        total_loss = 0.0
        all_logits = []
        all_targets = []
        all_filenames = []

        running_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Eval after {self.cur_epoch} - Loss : {running_loss}"):
                audio, labels, filenames = batch
                audio = audio.to(self.device)
                labels = labels.to(self.device)
                
                # if isinstance(self.pipeline, FilterClassifier):
                #     if self.cur_epoch < half_epochs:
                #         logits, wav = self.pipeline(audio, use_post_filter=False)
                #     else:
                #         logits, wav = self.pipeline(audio, use_post_filter=True)
                # else:
                outs = self.pipeline(audio, eval_mode = True)
                logits, wav, _, _ = outs
                
                if self.train_mode == "multi-label" :
                    labels = labels.float()
                    logits = torch.sigmoid(logits)
                

                loss = self.criterion(logits, labels)
                    
                # if self.cur_epoch == self.cfg.train.epochs or self.cfg.return_wav and self.cfg.mode=='only_eval': 
                #     self.save_batch_to_wav(wav, filenames)
                
                
                running_loss = loss.item()
                total_loss += loss.item()

                all_logits.append(logits.cpu())
                all_targets.append(labels.cpu())
                all_filenames.extend(filenames)
            

        all_logits = torch.cat(all_logits, dim=0)    # [N, C]
        all_targets = torch.cat(all_targets, dim=0)  # [N, C]
        
        
        
        # Compute mAP
        if self.train_mode == "multi-label" : 
            
            y_true = all_targets.numpy()
            y_pred = all_logits.numpy()
            mAP = average_precision_score(
                y_true,
                y_pred,
                average='macro')
            
            print(f"mAP: {mAP:.4f}")
        
        elif self.train_mode == "multi-class" :
        
            y_true = all_targets.numpy()
            y_pred = all_logits.argmax(axis=1).numpy()
            

            
        # Accuracy
        
        acc = accuracy_score(
            y_true, y_pred)

        # F1 scores
        f1_macro = f1_score(
            y_true, y_pred, average='macro')  # good for class imbalance
        f1_micro = f1_score(
            y_true, y_pred, average='micro')  # good for overall precision/recall

        print(f"Accuracy: {acc:.4f}, F1 (macro): {f1_macro:.4f}, F1 (micro): {f1_micro:.4f}, Validation Loss: {total_loss / len(self.val_loader):.4f}")
        
        
        if self.cfg.log.wandb : 
            wandb.log({
                "val/loss": total_loss / len(self.val_loader),
                "val/accuracy": acc,
                "val/f1_macro": f1_macro,
                "val/f1_micro": f1_micro,
                "epoch": self.cur_epoch
            })
        

        records = []
        for fname, true_label, pred_label in zip(all_filenames, y_true, y_pred):
            records.append({
                "filename": fname,
                "true_label": int(true_label),
                "predicted_label": int(pred_label),
                "correct": int(true_label == pred_label)
            })
        df = pd.DataFrame(records)

        csv_dir = os.path.join(self.cfg.save_wav_dir, f"fold_{self.cfg.data.test_fold}")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"eval_results_epoch{self.cur_epoch}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved evaluation results to {csv_path}")
                

    def save_checkpoint(self, name=None) :
        if name is None :
            name = f"ep_{self.cur_epoch}"

        save_path = os.path.join(self.cfg.save_model_dir, f"fold_{self.cfg.data.test_fold}", name)
        os.makedirs(save_path, exist_ok=True)
        torch.save({
            'cur_epoch' : self.cur_epoch,
            'model_state_dict' : self.pipeline.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'cfg' : self.cfg
            }, save_path+'.pt')
        print({f"{save_path}.pt is SAVED!!"})

    def load_pretrained(self, path):
        print(f"Loading pretrained model from {path}")
        
        if path.endswith('.pt') :
            checkpoint = torch.load(path, map_location=self.device)
        else :
            checkpoint = torch.load(os.path.join(path, f"fold_{self.cfg.data.test_fold}", "latest.pt"), map_location=self.device)
        
        self.pipeline.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    def load_checkpoint(self, path):
        print(f"Loading checkpoint from {path}")
        
        if path.endswith('.pt') :
            checkpoint = torch.load(path, map_location=self.device)
        else :
            checkpoint = torch.load(os.path.join(path, f"fold_{self.cfg.data.test_fold}", "latest.pt"), map_location=self.device)
        
        self.pipeline.load_state_dict(checkpoint['model_state_dict'])
        if hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.cfg = checkpoint['cfg']
        self.cur_ckpt = checkpoint['cur_epoch']
        print(f"Loaded checkpoint from {path} (epoch {self.cur_epoch})")
        
    def save_batch_to_wav(self, tensor_batch, filenames, prefix='output'):
        """
        Save a batch of waveforms to .wav files.

        Args:
            tensor_batch (torch.Tensor): shape [B, 1, T]
            folder_path (str): directory to save wav files
            sample_rate (int): target sample rate
            prefix (str): filename prefix
        """
        folder_path = self.cfg.save_wav_dir
        os.makedirs(self.cfg.save_wav_dir, exist_ok=True)
        
        B = tensor_batch.size(0)
        
        for i in range(B):
            waveform = tensor_batch[i]  # shape [1, T]
            
            # Ensure shape is [channels, time]
            if waveform.dim() == 2 and waveform.shape[0] == 1:
                waveform = waveform.squeeze(0)  # [T]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # [1, T]
            
            filename = filenames[i].split('/')[-1]
            filename = filename+'.wav'
            filepath = os.path.join(folder_path, filename)
            torchaudio.save(filepath, waveform.cpu(), sample_rate=self.cfg.data.sr)

        print(f"Saved {B} files to {folder_path}")
    
    def sample_steps(self, file, save_path=None) :
        
        self.pipeline.eval()
        
        if save_path is None :
            save_path = os.path.join(self.cfg.save_wav_dir, "sampled")
            os.makedirs(save_path, exist_ok=True)
            
        waveform, sr = torchaudio.load(file)
        
        if sr != self.cfg.data.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.cfg.data.sr)
        
        out_dict = self.pipeline.sample_wav(waveform.to(self.device))
        
        for key, waveform in out_dict.items() : 
            audio_path = os.path.join(save_path, key+"_"+file.split("/")[-1])

            assert len(waveform.shape) == 2, f"must be 2D {waveform.shape}"
            torchaudio.save(audio_path, waveform.cpu(), sample_rate=self.cfg.data.sr)
            print(f"audio saved {audio_path}")

        
        
    
    
if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default="configs/classification_Opus_ESC-50.yaml")
    parser.add_argument("--test_fold", type=int, default=None)
    parser.add_argument("--bitrate", type=float, default=None)
    args = parser.parse_args()
    
    # predict the classification probability of each class

    cfg = OmegaConf.load(args.configs)
    
    if args.test_fold is not None :
        cfg.data.test_fold = args.test_fold 
    
    if args.bitrate is not None :
        cfg.baseline.bitrate = args.bitrate
    
    trainer = BeatsTrainer(cfg)
    
    if cfg.mode == "sample" :
        audio_path = "/data/ESC-50-master/audio/5-260164-A-23.wav"
        trainer.load_pretrained(cfg.load_model_dir)
        trainer.sample_steps(audio_path)
        
    elif cfg.mode == "only_eval" or cfg.mode== "eval_only" :
        trainer.load_pretrained(cfg.load_model_dir) 
        trainer.eval()
        # trainer.sample_steps(audio_path)
    else :
        trainer.train()
