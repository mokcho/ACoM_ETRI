import torch
import torch.nn as nn
import torchaudio
from transformers import EncodecModel

class ROITransformerMasker(nn.Module):
    def __init__(self, d_input=80, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, 1)  # ROI mask score
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        x: (B, T, F) - e.g., log-mel spectrogram
        Returns: (B, T, F) - masked spectrogram
        """
        B, T, F = x.shape
        x_proj = self.input_proj(x)       # (B, T, d_model)
        x_proj = x_proj.permute(1, 0, 2)  # (T, B, d_model)
        x_enc = self.transformer(x_proj)  # (T, B, d_model)
        x_enc = x_enc.permute(1, 0, 2)    # (B, T, d_model)
        
        mask_scores = self.output_proj(x_enc)     # (B, T, 1)
        mask = self.sigmoid(mask_scores)          # (B, T, 1)
        masked_x = x * mask                       # Apply mask per frame
        return masked_x, mask

class ROIConvTransformerAutoencoder(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=128, transformer_layers=2, kernel_size=9, stride=2):
        super().__init__()

        # Encoder: downsample with stride
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//4),  # [B, 128, 8000]
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),      # [B, 128, 4000]
            nn.GELU()
        )

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2),
            num_layers=transformer_layers
        )

        # Decoder: upsample with transposed conv
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=6, stride=6),  # [B, 128, 12000]
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim, input_channels, kernel_size=5, padding=2),  # [B, 1, 24000]
            nn.Tanh()  # Waveform range [-1, 1]
        )

    def forward(self, x):
        # x: [B, C, T]
        # x = torch.unsqueeze(x, dim=0) if len(x.shape) == 1 else torch.unsqueeze(x, dim=1) # [T] - >[1, T] OR [B, T] -> [B, 1, T]
        
        x = self.encoder(x)  # [B, 128, T/4]

        # Transformer expects: [T, B, C]
        x = x.permute(2, 0, 1)  # → [T, B, C]
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # → [B, C, T]

        x = self.decoder(x)  # [B, 1, 24000]
        return x

# Example usage
if __name__ == "__main__":
    B, C, T = 4, 1, 16000  # 4 examples, mono, 1 second at 24kHz
    dummy_audio = torch.randn(B, C, T)

    model = ROIConvTransformerAutoencoder()
    output = model(dummy_audio)

    print("Input shape :", dummy_audio.shape)
    print("Output shape:", output.shape)