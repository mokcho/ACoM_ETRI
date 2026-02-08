import os
import torch
import csv
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import random

import warnings


from tqdm import tqdm

def to_mono(mixture, random_ch=False):
    if mixture.ndim > 1:  # multi channel
        if not random_ch:
            mixture = torch.mean(mixture, 0)
        else:  # randomly select one channel
            indx = np.random.randint(0, mixture.shape[0] - 1)
            mixture = mixture[indx]
    return mixture


def pad_audio(audio, target_len, fs, test=False):
    if audio.shape[-1] < target_len:
        audio = torch.nn.functional.pad(
            audio, (0, target_len - audio.shape[-1]), mode="constant"
        )

        padded_indx = [target_len / len(audio)]
        onset_s = 0.000

    elif len(audio) > target_len:
        if test:
            clip_onset = 0
        else:
            clip_onset = random.randint(0, len(audio) - target_len)
        audio = audio[clip_onset : clip_onset + target_len]
        onset_s = round(clip_onset / fs, 3)

        padded_indx = [target_len / len(audio)]
    else:
        onset_s = 0.000
        padded_indx = [1.0]

    offset_s = round(onset_s + (target_len / fs), 3)
    return audio, onset_s, offset_s, padded_indx


def process_labels(df, onset, offset):
    df["onset"] = df["onset"] - onset
    df["offset"] = df["offset"] - onset
    df["onset"] = df.apply(lambda x: max(0, x["onset"]), axis=1)
    df["offset"] = df.apply(lambda x: min(10, x["offset"]), axis=1)

    df_new = df[(df.onset < df.offset)]
    return df_new.drop_duplicates()


def read_audio(file, multisrc, random_channel, pad_to, test=False):
    mixture, fs = torchaudio.load(file)

    if not multisrc:
        mixture = to_mono(mixture, random_channel)

    if pad_to is not None:
        mixture, onset_s, offset_s, padded_indx = pad_audio(mixture, pad_to, fs, test=test)
    else:
        padded_indx = [1.0]
        onset_s = None
        offset_s = None

    mixture = mixture.float()
    return mixture, onset_s, offset_s, padded_indx

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
            print("No File at", path)
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
    
class StronglyAnnotatedSet(Dataset):
    def __init__(
        self,
        audio_folder,
        tsv_entries,
        encoder,
        pad_to=10,
        fs=16000,
        return_filename=False,
        random_channel=False,
        multisrc=False,
        feats_pipeline=None,
        embeddings_hdf5_file=None,
        embedding_type=None,
        mask_events_other_than=None,
        test=False,
    ):
        self.encoder = encoder
        self.fs = fs
        self.pad_to = pad_to * fs
        self.return_filename = return_filename
        self.random_channel = random_channel
        self.multisrc = multisrc
        self.feats_pipeline = feats_pipeline
        self.embeddings_hdf5_file = embeddings_hdf5_file
        self.embedding_type = embedding_type
        self.test = test

        # we mask events that are incompatible with the current setting
        if mask_events_other_than is not None:
            # fetch indexes to mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))
            for indx, cls in enumerate(encoder.labels):
                if cls not in mask_events_other_than:
                    # set to zero corresponding entry, invalid class for this dataset
                    # we will skip loss computation
                    self.mask_events_other_than[indx] = 0
        else:
            # keep all, no mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))
        self.mask_events_other_than = self.mask_events_other_than.bool()
        assert embedding_type in [
            "global",
            "frame",
            None,
        ], "embedding type are either frame or global or None, got {}".format(
            embedding_type
        )

        tsv_entries = tsv_entries.dropna()

        examples = {}
        for i, r in tsv_entries.iterrows():
            if r["filename"] not in examples.keys():
                confidence = 1.0 if "confidence" not in r.keys() else r["confidence"]
                examples[r["filename"]] = {
                    "mixture": os.path.join(audio_folder, r["filename"]),
                    "events": [],
                    "confidence": confidence,
                }
                if not np.isnan(r["onset"]):
                    confidence = (
                        1.0 if "confidence" not in r.keys() else r["confidence"]
                    )
                    examples[r["filename"]]["events"].append(
                        {
                            "event_label": r["event_label"],
                            "onset": r["onset"],
                            "offset": r["offset"],
                            "confidence": confidence,
                        }
                    )
            else:
                if not np.isnan(r["onset"]):
                    confidence = (
                        1.0 if "confidence" not in r.keys() else r["confidence"]
                    )
                    examples[r["filename"]]["events"].append(
                        {
                            "event_label": r["event_label"],
                            "onset": r["onset"],
                            "offset": r["offset"],
                            "confidence": confidence,
                        }
                    )

        # we construct a dictionary for each example
        self.examples = examples
        self.examples_list = list(examples.keys())

        if self.embeddings_hdf5_file is not None:
            assert (
                self.embedding_type is not None
            ), "If you use embeddings you need to specify also the type (global or frame)"
            # fetch dict of positions for each example
            self.ex2emb_idx = {}
            f = h5py.File(self.embeddings_hdf5_file, "r")
            for i, fname in enumerate(f["filenames"]):
                self.ex2emb_idx[fname.decode("UTF-8")] = i
        self._opened_hdf5 = None

    def __len__(self):
        return len(self.examples_list)

    @property
    def hdf5_file(self):
        if self._opened_hdf5 is None:
            self._opened_hdf5 = h5py.File(self.embeddings_hdf5_file, "r")
        return self._opened_hdf5

    def __getitem__(self, item):
            # --- 1) Try to read audio; if missing/unreadable, fall back to silence ---
        
        c_ex = self.examples[self.examples_list[item]]
        missing_audio = False
        try:
            # quick existence check helps produce clearer error messages
            if not os.path.isfile(c_ex["mixture"]):
                raise FileNotFoundError(f"File not found: {c_ex['mixture']}")

            mixture, onset_s, offset_s, padded_indx = read_audio(
                c_ex["mixture"], self.multisrc, self.random_channel, self.pad_to, self.test,
            )
        
        except Exception as e :
            new_idx = torch.randint(0, self.__len__(), (1,)).item()
            return self.__getitem__(new_idx)
        
        # except Exception as e:
        #     missing_audio = True
        #     # warnings.warn(
        #     #     f"[Dataset] Audio missing or unreadable for '{c_ex.get('mixture','<unknown>')}'. "
        #     #     f"Using silence. Error: {repr(e)}"
        #     # )
        #     # Determine a sensible fallback length in samples
        #     # Prefer pad_to if you already use it; otherwise use sr * audio_max_len.
        #     sr = getattr(self, "sr", 16000)
        #     if getattr(self, "pad_to", None) is not None:
        #         target_len = int(self.pad_to)
        #     else:
        #         audio_max_len = getattr(self, "audio_max_len", 10)  # seconds
        #         target_len = int(sr * audio_max_len)

        #     # Make mono-silence [1, T] float32 (adjust if your pipeline expects a different shape)
        #     mixture = torch.zeros(1, target_len, dtype=torch.float32)

        #     # Reasonable time bounds for label processing
        #     onset_s = 0.0
        #     offset_s = target_len / float(sr)

        #     # If your code expects a real padded index, keep 0 as a safe default
        #     padded_indx = [0]
            
        # if mixture.dim() == 2:
        #     assert mixture.size(0) == 1 or mixture.size(1) == 1
        #     mixture = mixture.squeeze(0).squeeze(0)

        # labels
        labels = c_ex["events"]

        # to steps
        labels_df = pd.DataFrame(labels)
        labels_df = process_labels(labels_df, onset_s, offset_s)

        # check if labels exists:
        if not len(labels_df):
            max_len_targets = self.encoder.n_frames
            strong = torch.zeros(max_len_targets, len(self.encoder.labels)).float()
        else:
            strong = self.encoder.encode_strong_df(labels_df)
            strong = torch.from_numpy(strong).float()

        out_args = [mixture, strong.transpose(0, 1), padded_indx]

        # if self.feats_pipeline is not None:
        #     # use this function to extract features in the dataloader and apply possibly some data augm
        #     feats = self.feats_pipeline(mixture)
        #     out_args.append(feats)
        if self.return_filename:
            out_args.append(c_ex["mixture"])

        # if self.embeddings_hdf5_file is not None:
        #     name = Path(c_ex["mixture"]).stem
        #     index = self.ex2emb_idx[name]

        #     if self.embedding_type == "global":
        #         embeddings = torch.from_numpy(
        #             self.hdf5_file["global_embeddings"][index]
        #         ).float()
        #     elif self.embedding_type == "frame":
        #         embeddings = torch.from_numpy(
        #             np.stack(self.hdf5_file["frame_embeddings"][index])
        #         ).float()
        #     else:
        #         raise NotImplementedError

        #     out_args.append(embeddings)

        # if self.mask_events_other_than is not None:
        #     out_args.append(self.mask_events_other_than)
        
        # assert len(out_args) == 3, f"Unexpected number of output arguments on : {out_args}"
        

        return out_args
    
class WeakSet(Dataset):
    def __init__(
        self,
        audio_folder,
        tsv_entries,
        encoder,
        pad_to=10,
        fs=16000,
        return_filename=False,
        random_channel=False,
        multisrc=False,
        feats_pipeline=None,
        embeddings_hdf5_file=None,
        embedding_type=None,
        mask_events_other_than=None,
        test=False,
    ):
        self.encoder = encoder
        self.fs = fs
        self.pad_to = pad_to * fs
        self.return_filename = return_filename
        self.random_channel = random_channel
        self.multisrc = multisrc
        self.feats_pipeline = feats_pipeline
        self.embeddings_hdf5_file = embeddings_hdf5_file
        self.embedding_type = embedding_type
        self.mask_events_other_than = mask_events_other_than
        self.test = test

        if mask_events_other_than is not None:
            # fetch indexes to mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))
            for indx, cls in enumerate(encoder.labels):
                if cls not in mask_events_other_than:
                    # set to zero corresponding entry, invalid class for this dataset
                    # we will skip loss computation
                    self.mask_events_other_than[indx] = 0
        else:
            # keep all, no mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))

        self.mask_events_other_than = self.mask_events_other_than.bool()
        assert embedding_type in [
            "global",
            "frame",
            None,
        ], "embedding type are either frame or global or None, got {}".format(
            embedding_type
        )

        examples = {}
        for i, r in tsv_entries.iterrows():
            if r["filename"] not in examples.keys():
                examples[r["filename"]] = {
                    "mixture": os.path.join(audio_folder, r["filename"]),
                    "events": r["event_labels"].split(","),
                }

        self.examples = examples
        self.examples_list = list(examples.keys())

        if self.embeddings_hdf5_file is not None:
            assert (
                self.embedding_type is not None
            ), "If you use embeddings you need to specify also the type (global or frame)"
            # fetch dict of positions for each example
            self.ex2emb_idx = {}
            f = h5py.File(self.embeddings_hdf5_file, "r")
            for i, fname in enumerate(f["filenames"]):
                self.ex2emb_idx[fname.decode("UTF-8")] = i
        self._opened_hdf5 = None

    def __len__(self):
        return len(self.examples_list)

    @property
    def hdf5_file(self):
        if self._opened_hdf5 is None:
            self._opened_hdf5 = h5py.File(self.embeddings_hdf5_file, "r")
        return self._opened_hdf5

    def __getitem__(self, item):
        file = self.examples_list[item]
        c_ex = self.examples[file]

        mixture, _, _, padded_indx = read_audio(
            c_ex["mixture"], self.multisrc, self.random_channel, self.pad_to, self.test
        )

        # labels
        labels = c_ex["events"]
        # check if labels exists:
        max_len_targets = self.encoder.n_frames
        weak = torch.zeros(max_len_targets, len(self.encoder.labels))
        if len(labels):
            weak_labels = self.encoder.encode_weak(labels)
            weak[0, :] = torch.from_numpy(weak_labels).float()

        out_args = [mixture, weak.transpose(0, 1), padded_indx]

        if self.feats_pipeline is not None:
            feats = self.feats_pipeline(mixture)
            out_args.append(feats)

        if self.return_filename:
            out_args.append(c_ex["mixture"])

        if self.embeddings_hdf5_file is not None:
            name = Path(c_ex["mixture"]).stem
            index = self.ex2emb_idx[name]

            if self.embedding_type == "global":
                embeddings = torch.from_numpy(
                    self.hdf5_file["global_embeddings"][index]
                ).float()
            elif self.embedding_type == "frame":
                embeddings = torch.from_numpy(
                    np.stack(self.hdf5_file["frame_embeddings"][index])
                ).float()
            else:
                raise NotImplementedError

            out_args.append(embeddings)

        if self.mask_events_other_than is not None:
            out_args.append(self.mask_events_other_than)

        return out_args
