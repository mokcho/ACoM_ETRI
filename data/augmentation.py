import torch
import torch.nn as nn
import torchaudio.functional as F
import torchaudio.transforms as Tr
import random


class RandomVol(nn.Module):
    def __init__(self, min_gain=0.5, max_gain=1.5, p=1.0):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.p = p

    def forward(self, waveform):
        if random.random() < self.p:
            gain = random.uniform(self.min_gain, self.max_gain)
            return waveform * gain
        return waveform


class AddGaussianNoise(nn.Module):
    def __init__(self, min_std=0.001, max_std=0.01, p=1.0):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std
        self.p = p

    def forward(self, waveform):
        if random.random() < self.p:
            std = random.uniform(self.min_std, self.max_std)
            noise = torch.randn_like(waveform) * std
            return waveform + noise
        return waveform


class RandomFade(nn.Module):
    def __init__(self, min_len=50, max_len=400, p=0.5):
        super().__init__()
        self.min_len = min_len
        self.max_len = max_len
        self.p = p

    def forward(self, waveform):
        if random.random() < self.p:
            T = waveform.shape[-1]
            fade_in_len = min(int(random.uniform(self.min_len, self.max_len)), T // 2)
            fade_out_len = min(int(random.uniform(self.min_len, self.max_len)), T // 2)
            # return T.Fade(waveform, fade_in_len=fade_in_len, fade_out_len=fade_out_len)
            return Tr.Fade(fade_in_len=fade_in_len, fade_out_len=fade_out_len, fade_shape="linear")(waveform)
        return waveform


class WaveformAugmentations(nn.Module):
    def __init__(self, apply_prob=0.5):
        super().__init__()
        self.apply_prob = apply_prob
        self.augmentations = nn.ModuleList([
            RandomVol(min_gain=0.7, max_gain=1.3, p=1.0),
            AddGaussianNoise(min_std=0.002, max_std=0.01, p=1.0),
            RandomFade(min_len=100, max_len=500, p=0.5),
        ])

    def forward(self, waveform):
        if random.random() > self.apply_prob:
            return waveform
        for aug in self.augmentations:
            if random.random() < 0.5:
                waveform = aug(waveform)
        return waveform
