import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from .base_model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        
        self.lstm = nn.LSTM(
            input_size=config['d_model'],
            hidden_size=config['hidden_size'],
            num_layers=config['n_layers'],
            dropout=config['dropout'] if config['n_layers'] > 1 else 0,
            batch_first=True,
            bidirectional=config.get('bidirectional', False)
        )
        
        # Account for bidirectional in output layer
        output_size = config['hidden_size'] * 2 if config.get('bidirectional', False) else config['hidden_size']
        self.output_layer = nn.Linear(output_size, config['vocab_size'])
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
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
            
    def forward(self, x, hidden=None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)
        
        # Embed input
        embedded = self.dropout(self.embedding(x))
        # embedded shape: (batch_size, sequence_length, d_model)
        
        # Initialize hidden state and cell state if not provided
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)
            
        # Process through LSTM
        output, (hidden_state, cell_state) = self.lstm(embedded, hidden)
        # output shape: (batch_size, sequence_length, hidden_size * num_directions)
        # hidden_state shape: (num_layers * num_directions, batch_size, hidden_size)
        # cell_state shape: (num_layers * num_directions, batch_size, hidden_size)
        
        # Apply dropout to output
        output = self.dropout(output)
        
        # Project to vocabulary size
        logits = self.output_layer(output)
        # logits shape: (batch_size, sequence_length, vocab_size)
        
        return logits, (hidden_state, cell_state)
    
    def _init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        num_directions = 2 if self.config.get('bidirectional', False) else 1
        return (
            torch.zeros(
                self.config['n_layers'] * num_directions,
                batch_size,
                self.config['hidden_size'],
                device=device
            ),
            torch.zeros(
                self.config['n_layers'] * num_directions,
                batch_size,
                self.config['hidden_size'],
                device=device
            )
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