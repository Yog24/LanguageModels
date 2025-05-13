import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base_model import BaseModel

class RNNModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        
        self.rnn = nn.RNN(
            input_size=config['d_model'],
            hidden_size=config['hidden_size'],
            num_layers=config['n_layers'],
            dropout=config['dropout'] if config['n_layers'] > 1 else 0,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(config['hidden_size'], config['vocab_size'])
        self.dropout = nn.Dropout(config['dropout'])
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, x, hidden=None):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)
        
        # Embed input
        embedded = self.dropout(self.embedding(x))
        # embedded shape: (batch_size, sequence_length, d_model)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)
            
        # Process through RNN
        output, hidden = self.rnn(embedded, hidden)
        # output shape: (batch_size, sequence_length, hidden_size)
        # hidden shape: (n_layers, batch_size, hidden_size)
        
        # Apply dropout to output
        output = self.dropout(output)
        
        # Project to vocabulary size
        logits = self.output_layer(output)
        # logits shape: (batch_size, sequence_length, vocab_size)
        
        return logits, hidden
    
    def _init_hidden(self, batch_size, device):
        return torch.zeros(
            self.config['n_layers'],
            batch_size,
            self.config['hidden_size'],
            device=device
        )
    
    def generate(self, prompt, max_length: int, temperature: float = 1.0):
        self.eval()
        with torch.no_grad():
            x = prompt
            hidden = None
            
            for _ in range(max_length):
                logits, hidden = self(x, hidden)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_token], dim=1)
                
        return x 