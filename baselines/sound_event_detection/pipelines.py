import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F

from baselines.beats.BEATs import BEATs, BEATsConfig
from baselines.beats.Tokenizers import TokenizersConfig, Tokenizers

from baselines.ast.src.models.ast_models import ASTModel
from encodec.model import EncodecModel

# class PostEncodecEnhancerSConv(nn.Module): 
#     def __init__(self, n_filters, channels, last_kernel_size, norm, norm_params, causal, pad_mode):
#         super().__init__()
        
#         self.extra_layer = nn.Sequential(
#             m.SConv1d(n_filters, channels, last_kernel_size, norm=norm, norm_kwargs=norm_params,
#                     causal=causal, pad_mode=pad_mode),
#             nn.Tanh()  # or any final activation
#         )

#     def forward(self, z):
#         x = self.extra_layer(z)
#         return x

# class FilterCRNN(nn.Module):
#     def __init__(self, front_layer_config, frozen_crnn):
#         super().__init__()
#         self.post_filter = PostEncodecEnhancerSConv(n_filters=1,
#                                                     channels=1,
#                                                     last_kernel_size=self.encodec.decoder.last_kernel_size,
#                                                     norm='none',
#                                                     norm_params={},
#                                                     causal=False,
#                                                     pad_mode=self.encodec.decoder.pad_mode)
#         self.crnn = frozen_crnn  # frozen crnn

#     def forward(self, x):
#         x = self.post_filter(x)
#         return self.crnn(x)

class BaseEventDetector(nn.Module):
    """
    Base event detector using BEATs model
    Outputs frame-level predictions for strong labels or clip-level for weak labels
    """
    def __init__(self, cfgs, num_classes, mode='weak'):
        super().__init__()
        self.cfg = cfgs
        self.mode = mode  # 'strong' or 'weak'
        self.num_classes = num_classes

        if self.cfg.baseline.model.lower() == 'beats':
            checkpoint = torch.load('/data/BEATs/BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt1.pt')
            checkpoint['cfg']['finetuned_model'] = False
            cfg = BEATsConfig(checkpoint['cfg'])

            self.baseline = BEATs(cfg)
            self.baseline.load_state_dict(checkpoint['model'], strict = False)
            
            self.embed_dim = cfg.encoder_embed_dim
        
        elif self.cfg.baseline.model.lower() == 'ast': 
            
            # Mel-spectrogram transform (AST uses this)
            self.mel_trans = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                hop_length=160,
                n_mels=128,
                center=True,        # ← Important!
                pad_mode='reflect'  # ← Default, but explicit
            )
            
            # Initialize AST model
            self.baseline = ASTModel(
                label_dim=527,
                input_tdim=1214,  # ← ESC-50 uses 512, not 1024!
                input_fdim=128,
                fstride=10,
                tstride=10,
                audioset_pretrain=True  # ← Loads AudioSet, auto-resizes pos_embed
            )
            
            #Remove AST built in classifier
            self.baseline.mlp_head = nn.Identity() 
            
            self.embed_dim = 768  # AST uses ViT-Base which has 768-dim embeddings
            
        
        # Freeze BEATs if needed
        for param in self.baseline.parameters():
            param.requires_grad = False
        
        # Get embedding dimension from BEAT
        
        # Event detection head
        # if mode == 'strong':
        # Frame-level prediction head
        # self.detection_head = nn.Sequential(
        #     nn.LayerNorm(self.embed_dim),
        #     nn.Linear(self.embed_dim, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(512, num_classes)
        # )
        # else:  # weak
            # Clip-level prediction head with temporal pooling
        
        self.detection_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, audio, eval_mode=False, padding_mask=None):
        """
        Args:
            audio: [B, 1, T] or [B, T]
        Returns:
            logits: [B, T_frames, C] for strong, [B, C] for weak
            audio: passthrough
            features: BEATs features
            orig_features: same as features (for compatibility)
        """
        # Ensure audio is [B, T]
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        processed_input = audio
            
        if self.cfg.baseline.model.lower() == 'beats':
            
        
            # Extract BEATs features
            with torch.set_grad_enabled(not eval_mode or self.cfg.baseline.get('finetune_beats', False)):
                features = self.baseline.extract_features(audio)[0]  # [B, T_frames, D]
                
            features = torch.mean(features, dim=1)
        
        elif self.cfg.baseline.model.lower() == 'ast':
            # Debug: print actual audio length
            # print(f"Input audio shape: {audio.shape}")
            
            target_length = 194240  # For 1024 frames
            if audio.shape[-1] < target_length:
                audio = F.pad(audio, (0, target_length - audio.shape[-1]))
            else:
                audio = audio[:, :target_length]
            
            # print(f"After padding: {audio.shape}")
            
            # Mel-spectrogram
            mel = self.mel_trans(audio)
            # print(f"Mel shape: {mel.shape}")  # Check actual frames produced
            
            log_mel = (mel + 1e-6).log()
            log_mel = (log_mel + 4.2677393) / 4.5689974
            log_mel = log_mel.transpose(1, 2)
            
            # print(f"Log-mel shape before AST: {log_mel.shape}") #1214 128
            
            features = self.baseline(log_mel) #1214 768
            
        
        # Detection head
        if self.mode == 'strong':
            raise NotImplementedError # Frame-level predictions
            logits = self.detection_head(features)  # [B, T_frames, C]
        else:  # weak
            logits = self.detection_head(features)
        
        return logits, processed_input, features, features

class EnCodecEventDetector(BaseEventDetector):
    def __init__(self, cfgs, num_classes, mode='weak'):
        super().__init__(cfgs, num_classes, mode='weak')
        
        
        self.encodec = EncodecModel.encodec_model_24khz()
        
        if self.cfg.baseline.bitrate is not None :
            self.bitrate = self.cfg.baseline.bitrate
            print(f"setting encodec bitrate to {self.bitrate}")
            self.encodec.set_target_bandwidth(self.bitrate)
        
        for param in self.encodec.parameters():
            param.requires_grad=False
        
        if self.cfg.baseline.freeze_classifier :
            print("freezing classifier")
            for param in self.detection_head.parameters():
                param.requires_grad=False
                
    
    def forward(self, audio, eval_mode=False, padding_mask=None):
        # print(audio.shape) # for some reason, its [B, T]
        audio = audio.unsqueeze(1) # make it [B, 1, T]
        x = self.encodec(audio)

        # squeeze 1
        x= x.squeeze(1)
        return super().forward(x, eval_mode=eval_mode) 
    
class FilterEventDetector(BaseEventDetector):
    def __init__(self, cfgs, num_classes, mode='weak'):
        super().__init__(cfgs, num_classes, mode='weak')
        
        
        self.encodec = EncodecModel.encodec_model_24khz()
        
        if self.cfg.baseline.bitrate is not None :
            self.bitrate = self.cfg.baseline.bitrate
            print(f"setting encodec bitrate to {self.bitrate}")
            self.encodec.set_target_bandwidth(self.bitrate)
        
        for param in self.encodec.parameters():
            param.requires_grad=False
        
        if self.cfg.baseline.freeze_classifier :
            print("freezing classifier")
            for param in self.detection_head.parameters():
                param.requires_grad=False
                
        self.post_filter = PostEncodecEnhancer1D(in_channels=1, num_blocks=cfgs.data.post_blocks)
    
    
    def forward(self, audio, eval_mode=False):
        # print(audio.shape) # for some reason, its [B, T]
        audio = audio.unsqueeze(1) # make it [B, 1, T]
        x = self.encodec(audio)
        x = self.post_filter(x)  # your pre-processing

        # squeeze 1
        x= x.squeeze(1)
        return super().forward(x, eval_mode=eval_mode) 
        
