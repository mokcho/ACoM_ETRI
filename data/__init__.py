from .dataset import *
from .augmentation import *
import torch.nn.functional as F


def collate_fn_multi_label(batch):
    audios = [item['audio'] for item in batch]
    labels = [item['label'] for item in batch]
    
    lengths = [a.shape[-1] for a in audios]
    max_len = max(lengths)

    padded = [F.pad(a, (0, max_len - a.shape[-1])) for a in audios]
    batch_audio = torch.stack(padded)
    batch_labels = torch.stack(labels)
    
    return batch_audio, batch_labels #, batch_paths

def collate_fn_multi_class(batch):
    audios = [item['audio'] for item in batch]
    
    lengths = [a.shape[-1] for a in audios]
    max_len = max(lengths)

    padded = [F.pad(a, (0, max_len - a.shape[-1])) for a in audios]
    batch_audio = torch.stack(padded)
    batch_labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)  # shape: [B]
    batch_paths = [item['filename'] for item in batch]
    
    return batch_audio, batch_labels, batch_paths