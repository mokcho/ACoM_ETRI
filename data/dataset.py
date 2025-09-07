import os
import torch
import csv
from torch.utils.data import Dataset
import torchaudio
import pandas as pd

from tqdm import tqdm

class AudioSetDataset(Dataset):
    def __init__(self, root_dir, annotation_dir, extension=".wav", sample_rate=16000, max_duration=10):
        self.file_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(root_dir)
            for file in files if file.endswith(extension)
        ]
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration) if max_duration else None
        
        self.annotation = self.open_annotation(annotation_dir)
        

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if self.max_samples:
            waveform = waveform[:, :self.max_samples]
            if waveform.shape[1] < self.max_samples:
                pad_len = self.max_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        return waveform, path, self.get_label(path)
    
    def open_annotation(self, annotation_dir):
        with open(annotation_dir, newline='') as file :
            reader = csv.reader(file)
            next(reader)
            next(reader)
            label_dict = {}
            for i, row in tqdm(enumerate(reader), desc=f"Reading {os.path.basename(annotation_dir)}", unit="clip") :
                label_dict[row[0]] = row[3:]
        
        return label_dict
    
    def get_label(self, youtube_id) :
        youtube_id = os.path.basename(youtube_id).replace('.wav', '')
        return self.annotation[youtube_id]

class ESCDataset(Dataset):
    def __init__(self,root_dir, annotation_dir, label2id = None, sample_rate=16000, test_fold=1, train=True):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.annotation = self.open_annotation(annotation_dir)
        self.annotation = self.get_fold_split(test_fold=test_fold, train=train)
        
        all_labels = self.annotation['category'].tolist()
        unique_labels = sorted(set(all_labels))  # all_labels is a list of all text labels
        
        if label2id is not None : 
            self.label2id = label2id
        
        else : 
            self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx) :
        
        try :
            while True : 
                
                row = self.annotation.iloc[idx]
                filename = row['filename']
                path = os.path.join(self.root_dir, filename)
                waveform, sr = torchaudio.load(path)
            
                break
                    
                
        except Exception as e :
            new_idx = torch.randint(0, self.__len__(), (1,)).item()
            return self.__getitem__(new_idx)
        
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        label = self.get_label(path)
        label_id = self.label2id[label]  # convert to integer
        
        # multi_hot = torch.zeros(len(self.label2id))
        # multi_hot[label_id] = 1.0
        
        output = {
            'audio' : waveform,
            'filename' : path,
            'label' : label_id
        }
        return output
    
    def open_annotation(self, annotation_dir):
        df = pd.read_csv(annotation_dir)
        return df
    
    def get_fold_split(self, test_fold=1, train=True):
        df = self.annotation
        if train : 
            out_df = df[df['fold'] != test_fold].reset_index(drop=True)
        else : 
            out_df = df[df['fold'] == test_fold].reset_index(drop=True)
        return out_df

    
    def get_label(self, path):
        filename = os.path.basename(path)
        label_series = self.annotation.loc[self.annotation['filename'] == filename, 'category']
        
        if label_series.empty:
            raise ValueError(f"Label not found for file: {filename}")
        
        return label_series.iloc[0]  # return string
        