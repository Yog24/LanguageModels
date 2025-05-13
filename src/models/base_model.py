import torch
import torch.nn as nn
from typing import Dict, Any

class BaseModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
    def forward(self, x):
        raise NotImplementedError
        
    def generate(self, prompt, max_length: int, temperature: float = 1.0):
        raise NotImplementedError
        
    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        
    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> 'BaseModel':
        """Factory method to create models"""
        from .gpt_model import GPTModel
        from .transformer_model import TransformerModel
        from .rnn_model import RNNModel
        from .lstm_model import LSTMModel
        
        model_map = {
            'gpt': GPTModel,
            'transformer': TransformerModel,
            'rnn': RNNModel,
            'lstm': LSTMModel
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return model_map[model_type](config) 