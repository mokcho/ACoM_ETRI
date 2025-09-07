import torch
import torch.nn as nn
import torchaudio
import wandb

from baselines.beats.BEATs import BEATs, BEATsConfig
from baselines.beats.Tokenizers import TokenizersConfig, Tokenizers
from encodec.model import EncodecModel

from networks.roi_transformer import ROIConvTransformerAutoencoder
from networks.post_encodec import *


class BaseClassifier(nn.Module):
    def __init__(self, cfgs, label2id) :#frontend, beats_model, num_classes, freeze_beats=True):
        super().__init__()  
        self.cfgs = cfgs
        
        if self.cfgs.baseline.model.lower() == 'beats' :
            self.model_ckpt = torch.load('/data/BEATs/BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt1.pt') #(self.cfgs.model_ckpt) 
            self.model_ckpt['cfg']['finetuned_model'] = False
            self.baseline = BEATs(BEATsConfig(self.model_ckpt['cfg']))
            if not self.cfgs.baseline.freeze_classifier :
                print("Loading BEATs model from /data/BEATs/BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt1.pt")
                self.baseline.load_state_dict(self.model_ckpt['model'], strict = False)
        else :
            print(f"BASELINE {self.cfgs.baseline} is NOT SUPPORTED")
            
        if self.cfgs.baseline.freeze_baseline :
            print(f"freezing {self.cfgs.baseline.model}")
            for param in self.baseline.parameters():
                param.requires_grad=False
        else :
            print("WARNING : baseline is training")
            tokenizer_ckpt = torch.load('/data/BEATs/Tokenizer_iter3_plus_AS20K.pt')
            tokenizers_cfg = TokenizersConfig(tokenizer_ckpt['cfg'])
            self.tokenizer = Tokenizers(tokenizers_cfg)
            self.tokenizer.load_state_dict(tokenizer_ckpt['model'])
        
        num_classes = len(label2id)

        if self.cfgs.train_classifier : 
            self.classifier = nn.Sequential(
                nn.Linear(self.baseline.cfg.encoder_embed_dim, 256),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(256, num_classes)
                
            )

    def forward(self, x, eval_mode=None): 
        
        # print(x.shape) # [B, 1, T]
        x = x.squeeze(1)
        features = self.baseline.extract_features(x, padding_mask=None)[0] 
        
        
        if self.cfgs.train_classifier :  
            
            logits = self.classifier(features)
            # print(logits.shape) # batch_size, features, num_classes
            logits = logits.mean(dim=1)
            output = logits

            
        else :
            output = features, None


        return output, x, features, None
    
    
class EnCodecClassifier(BaseClassifier) :
    def __init__(self, cfgs, label2id):
        super().__init__(cfgs, label2id)
        

        self.encodec = EncodecModel.encodec_model_24khz()
        
        if self.cfgs.baseline.bitrate is not None :
            self.bitrate = self.cfgs.baseline.bitrate
            print(f"setting encodec bitrate to {self.bitrate}")
            self.encodec.set_target_bandwidth(self.bitrate)
        
        if self.cfgs.baseline.freeze_encodec :
            print("freezing encodec")
            for param in self.encodec.parameters():
                param.requires_grad=False
                
        self.encodec.train()
        self.encodec.quantizer.vq.training = True
        
        if self.cfgs.baseline.freeze_classifier :
            print("freezing classifier")
            for param in self.classifier.parameters():
                param.requires_grad=False
        
    def forward(self, x, use_post_filter=True, eval_mode=False):


        x = self.encodec(x) #input should be [B, 1, T]
        x = x.squeeze(1) # [B, T]

        features = self.baseline.extract_features(x, padding_mask=None)[0]  # [B, T, D]

        if self.cfgs.train_classifier:
            logits = self.classifier(features)   # [B, T, C]
            logits = logits.mean(dim=1)         # [B, C]
            

            return logits, x, features, None
        
        else:
            return features, x

class FilterClassifier(EnCodecClassifier) :
    
    # TODO backprop through encodec
    def __init__(self, cfgs, label2id) :
        super().__init__(cfgs, label2id)
        # self.pre_filter = PostEncodecEnhancer1D(in_channels=1, num_blocks=cfgs.train.post_blocks)
        self.post_filter = PostEncodecEnhancer1D(in_channels=1, num_blocks=cfgs.train.post_blocks)                                          
    
    def forward(self, x, use_post_filter=True, eval_mode=False):
        """
        Args:
            x: input tensor [B, 1, T]
            use_post_filter (bool): whether to use the post_filter module. If False, skip post_filter.
        Returns:
            logits, x (or features, x if not training classifier)
        """
        # pre_filtered = self.pre_filter(x)
        # print("After pre-filter, input to encodec", x.shape) # x is [B, 1, T]
        
        # if eval_mode :  #training without quantization
        # x = self.encodec(x)
            
        # else : # training without quant, but latents
        # z = self.encodec.encode(pre_filtered)
            
        # x = self.encodec.decode(z)[:, :, :pre_filtered.shape[-1]]
        # else :    # training with quant

        # z     = self.encodec.encoder(pre_filtered)          # (B, C, T')
        # print(z.shape) # B, 128, 375
        
        # mid filter
        # z = self.pre_filter(z)
        
        ### Mid Filter
        # nn.Linear expects (batch * time, C) so we reshape:
        # B, C, T_ = z.shape                # C should equal latent_dim
        # z = z.permute(0, 2, 1)            # (B, T', C)
        # z = self.pre_filter(z)            # (B, T', C)
        # z = F.relu(z)                     # optional non-linearity
        # z = z.permute(0, 2, 1)            # (B, C, T') back to conv layout
        # ###
        
        # qr    = self.encodec.quantizer(z,
        #                                self.encodec.frame_rate,
        #                                self.encodec.bandwidth)  
        # print(qr)
        # if use_post_filter : 
        # z_q      = qr.quantized# STE happens here
        # else :
            # z_q = z
        
        # B, C, T_ = z_q.shape                # C should equal latent_dim
        
        # z_q = z_q.permute(0, 2, 1)            # (B, T', C)
        # z_q = self.pre_filter(z_q)            # (B, T', C)
        # z_q = F.relu(z_q)                     # optional non-linearity
        # z_q = z_q.permute(0, 2, 1)            # (B, C, T') back to conv layout
        ###
        
        # x  = self.encodec.decoder(z_q)
        orig_x = x
        
        
        x = self.encodec(x)
        # if use_post_filter:
        x = self.post_filter(x) 
            
        x = x.squeeze(1)
        orig_x = orig_x.squeeze(1)
        # print("Input to Baseline", x.shape) # x is [B, T]
        assert orig_x.shape == x.shape, f"Shape mismatch: orig_x {orig_x.shape} != x {x.shape}"
        
        if not eval_mode :
            orig_features = self.baseline.extract_features(orig_x, padding_mask=None)[0]
        
        else :
            orig_features = None
        
        features = self.baseline.extract_features(x, padding_mask=None)[0]  # input should be [B, T]
        
        if self.cfgs.train_classifier:
            logits = self.classifier(features)   # [B, T, C]
            logits = logits.mean(dim=1)         # [B, C]
            return logits, x, features, orig_features
        
        else:
            return (features, orig_features), x
        
    
    def sample_wav(self, wav, format="wav"):
        
        org_wav = wav
        
        org_wav = org_wav.unsqueeze(0)
        # print(org_wav.shape) 
        pre_filter_wav = self.post_filter(org_wav)
        encodec_wav = self.encodec(org_wav)
        
        # print(pre_filter_wav.shape)
        pre_encodec_wav = self.encodec(org_wav)
        # print(pre_encodec_wav.shape)
        
        post_filter_wav = self.post_filter(pre_encodec_wav)
        
        if format == 'wav':
        
            return {
                "original" : wav,
                "only-encodec" : encodec_wav.squeeze(0),
                "after_pre-filter" : pre_filter_wav.squeeze(0),
                "after_pre-filter-encodec" : pre_encodec_wav.squeeze(0),
                "after_post-filter" : post_filter_wav.squeeze(0)
            }
        
        else : 
            print("NOT SUPPORTED YET")

class SNFilterClassifier(EnCodecClassifier) :
    
    # TODO backprop through encodec
    def __init__(self, cfgs, label2id) :
        super().__init__(cfgs, label2id)
        self.post_filter = PostEncodecEnhancerSConv(n_filters=1,
                                                    channels=1,
                                                    last_kernel_size=self.encodec.decoder.last_kernel_size,
                                                    norm='none',
                                                    norm_params={},
                                                    causal=False,
                                                    pad_mode=self.encodec.decoder.pad_mode)
    
    def forward(self, x, use_post_filter=True, eval_mode=False):
        """
        Args:
            x: input tensor [B, 1, T]
            use_post_filter (bool): whether to use the post_filter module. If False, skip post_filter.
        Returns:
            logits, x (or features, x if not training classifier)
        """
        # pre_filtered = self.pre_filter(x)
        pre_filtered = x
        # print("After pre-filter, input to encodec", x.shape) # x is [B, 1, T]
        
        # if eval_mode :  #training without quantization
        x = self.encodec(pre_filtered)
            
        # else : # training without quant, but latents
        #     z = self.encodec.encode(pre_filtered)
            
        #     x = self.decode(z)[:, :, :pre_filtered.shape[-1]]
        # else :    # training with quant

        # z     = self.encodec.encoder(pre_filtered)          # (B, C, T')
        # print(z.shape) # B, 128, 375
        
        # mid filter
        # z = self.pre_filter(z)
        
        ### Mid Filter
        # nn.Linear expects (batch * time, C) so we reshape:
        # B, C, T_ = z.shape                # C should equal latent_dim
        # z = z.permute(0, 2, 1)            # (B, T', C)
        # z = self.pre_filter(z)            # (B, T', C)
        # z = F.relu(z)                     # optional non-linearity
        # z = z.permute(0, 2, 1)            # (B, C, T') back to conv layout
        # ###
        
        # qr    = self.encodec.quantizer(z,
        #                                self.encodec.frame_rate,
        #                                self.encodec.bandwidth)  
        # print(qr)
        # if use_post_filter : 
        # z_q      = qr.quantized# STE happens here
        # else :
            # z_q = z
        
        # B, C, T_ = z_q.shape                # C should equal latent_dim
        
        # z_q = z_q.permute(0, 2, 1)            # (B, T', C)
        # z_q = self.pre_filter(z_q)            # (B, T', C)
        # z_q = F.relu(z_q)                     # optional non-linearity
        # z_q = z_q.permute(0, 2, 1)            # (B, C, T') back to conv layout
        ###
        
        # x  = self.encodec.decoder(z_q)
        
        
        # x = self.encodec(pre_filtered)
        # if use_post_filter:
        x = self.post_filter(x) 
            
        x = x.squeeze(1)
        # print("Input to Baseline", x.shape) # x is [B, T]
        
        features = self.baseline.extract_features(x, padding_mask=None)[0]  # input should be [B, T]

        if self.cfgs.train_classifier:
            logits = self.classifier(features)   # [B, T, C]
            logits = logits.mean(dim=1)         # [B, C]
            return logits, x
        
        else:
            return features, x
        
    
    def sample_wav(self, wav, format="wav"):
        
        org_wav = wav
        
        org_wav = org_wav.unsqueeze(0)
        # print(org_wav.shape) 
        pre_filter_wav = self.post_filter(org_wav)
        encodec_wav = self.encodec(org_wav)
        
        # print(pre_filter_wav.shape)
        pre_encodec_wav = self.encodec(org_wav)
        # print(pre_encodec_wav.shape)
        
        post_filter_wav = self.post_filter(pre_encodec_wav)
        
        if format == 'wav':
        
            return {
                "original" : wav,
                "only-encodec" : encodec_wav.squeeze(0),
                "after_pre-filter" : pre_filter_wav.squeeze(0),
                "after_pre-filter-encodec" : pre_encodec_wav.squeeze(0),
                "after_post-filter" : post_filter_wav.squeeze(0)
            }
        
        else : 
            print("NOT SUPPORTED YET")
