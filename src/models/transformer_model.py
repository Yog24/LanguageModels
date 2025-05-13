import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base_model import BaseModel

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config['d_model'], config['n_heads'], 
                                             dropout=config['dropout'])
        self.norm1 = nn.LayerNorm(config['d_model'])
        self.norm2 = nn.LayerNorm(config['d_model'])
        self.ff = nn.Sequential(
            nn.Linear(config['d_model'], config['d_ff']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_ff'], config['d_model'])
        )
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config['d_model'], config['n_heads'],
                                             dropout=config['dropout'])
        self.cross_attn = nn.MultiheadAttention(config['d_model'], config['n_heads'],
                                              dropout=config['dropout'])
        self.norm1 = nn.LayerNorm(config['d_model'])
        self.norm2 = nn.LayerNorm(config['d_model'])
        self.norm3 = nn.LayerNorm(config['d_model'])
        self.ff = nn.Sequential(
            nn.Linear(config['d_model'], config['d_ff']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_ff'], config['d_model'])
        )
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        self_attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        
        cross_attn_out, _ = self.cross_attn(x, memory, memory,
                                          attn_mask=memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x

class TransformerModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.src_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.tgt_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_encoding = self._create_positional_encoding(config['max_seq_length'],
                                                           config['d_model'])
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config['n_layers'])
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(config) for _ in range(config['n_layers'])
        ])
        
        self.output_layer = nn.Linear(config['d_model'], config['vocab_size'])
        self.dropout = nn.Dropout(config['dropout'])

    def _create_positional_encoding(self, max_len, d_model):
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)

    def _create_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        src_mask = None  # All tokens can attend to all positions in encoder
        tgt_mask = self._create_causal_mask(tgt.size(1)).to(tgt.device)
        
        # Encoder
        src = self.src_embedding(src) * torch.sqrt(torch.tensor(self.config['d_model']))
        src = src + self.pos_encoding[:, :src.size(1), :].to(src.device)
        src = self.dropout(src)
        
        enc_output = src
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
            
        # Decoder
        tgt = self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(self.config['d_model']))
        tgt = tgt + self.pos_encoding[:, :tgt.size(1), :].to(tgt.device)
        tgt = self.dropout(tgt)
        
        dec_output = tgt
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, tgt_mask)
            
        output = self.output_layer(dec_output)
        return output

    def generate(self, src, max_length: int, temperature: float = 1.0):
        self.eval()
        with torch.no_grad():
            # Encode source sequence
            src_mask = None
            src = self.src_embedding(src) * torch.sqrt(torch.tensor(self.config['d_model']))
            src = src + self.pos_encoding[:, :src.size(1), :].to(src.device)
            src = self.dropout(src)
            
            enc_output = src
            for enc_layer in self.encoder_layers:
                enc_output = enc_layer(enc_output, src_mask)
            
            # Initialize target sequence with start token
            tgt = torch.tensor([[self.config['start_token']]], device=src.device)
            
            for _ in range(max_length):
                tgt_mask = self._create_causal_mask(tgt.size(1)).to(tgt.device)
                
                tgt_emb = self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(self.config['d_model']))
                tgt_emb = tgt_emb + self.pos_encoding[:, :tgt.size(1), :].to(tgt.device)
                tgt_emb = self.dropout(tgt_emb)
                
                dec_output = tgt_emb
                for dec_layer in self.decoder_layers:
                    dec_output = dec_layer(dec_output, enc_output, tgt_mask)
                
                output = self.output_layer(dec_output[:, -1:])
                output = output / temperature
                probs = F.softmax(output, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), num_samples=1)
                
                if next_token.item() == self.config['end_token']:
                    break
                    
                tgt = torch.cat([tgt, next_token], dim=1)
            
        return tgt 