import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base_model import BaseModel

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['n_heads']
        self.head_dim = config['d_model'] // config['n_heads']
        self.d_model = config['d_model']
        
        self.q = nn.Linear(config['d_model'], config['d_model'])
        self.k = nn.Linear(config['d_model'], config['d_model'])
        self.v = nn.Linear(config['d_model'], config['d_model'])
        self.proj = nn.Linear(config['d_model'], config['d_model'])
        
        self.dropout = nn.Dropout(config['dropout'])
        
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        
        q = self.q(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.proj(out)
        
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config['d_model'], config['d_ff'])
        self.fc2 = nn.Linear(config['d_ff'], config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['d_model'])
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config['d_model'])
        self.ff = FeedForward(config)
        self.dropout = nn.Dropout(config['dropout'])
        
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x

class GPTModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.position_embedding = nn.Embedding(config['max_seq_length'], config['d_model'])
        
        self.blocks = nn.ModuleList([
            GPTBlock(config) for _ in range(config['n_layers'])
        ])
        
        self.ln_f = nn.LayerNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(self, x):
        device = x.device
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        # Create causal mask
        mask = torch.tril(torch.ones((t, t), device=device)).view(1, 1, t, t)
        
        # Embeddings
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(pos)
        x = token_embeddings + position_embeddings
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def generate(self, prompt, max_length: int, temperature: float = 1.0):
        self.eval()
        with torch.no_grad():
            x = prompt
            for _ in range(max_length):
                logits = self(x)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_token], dim=1)
        return x 