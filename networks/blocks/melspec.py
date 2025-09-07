import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F

class MelExtractor(nn.Module):
    def __init__(self, sr=16000, n_mels=64, n_fft=1024, hop_length=512, frames=5, power=2.0, fmin=None, fmax=None):
        super().__init__()
        self.frames = frames
        self.n_mels = n_mels
        self.hop_length = hop_length

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=None,
            f_min=fmin or 0.0,
            f_max=fmax or (sr // 2),
            pad=0,
            power=power,
            normalized=False,
            center=True,
            n_mels=n_mels
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, waveform):
        """
        waveform: [B, T]
        returns: [B * num_chunks, n_mels * frames]
        """
        # [B, F, T']
        mel = self.mel_transform(waveform)  # power or amplitude
        mel_db = self.db_transform(mel)     # log-scale

        # Now extract overlapping frame chunks
        B, F, T = mel_db.shape
        num_chunks = T - self.frames + 1

        chunks = []
        for i in range(num_chunks):
            chunk = mel_db[:, :, i:i+self.frames]     # [B, F, frames]
            chunk = chunk.reshape(B, -1)              # [B, F * frames]
            chunks.append(chunk)

        all_chunks = torch.cat(chunks, dim=0)         # [B * num_chunks, F * frames]
        return all_chunks