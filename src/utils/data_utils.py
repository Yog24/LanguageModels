"""Utility functions for data loading and processing."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
import json
import os

class TextDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_seq_length: int,
        tokenizer_path: Optional[str] = None,
        is_training: bool = True
    ):
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.is_training = is_training
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.texts = f.readlines()
            
        # Load or create tokenizer
        if tokenizer_path and os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'r') as f:
                self.vocab = json.load(f)
        else:
            self.vocab = self._build_vocab()
            if tokenizer_path:
                with open(tokenizer_path, 'w') as f:
                    json.dump(self.vocab, f, indent=2)
                    
        self.vocab_size = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from training data."""
        vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        token_freq = {}
        
        # Count token frequencies
        for text in self.texts:
            for token in text.strip().split():
                if token not in token_freq:
                    token_freq[token] = 0
                token_freq[token] += 1
                
        # Sort by frequency and add to vocab
        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        for token, _ in sorted_tokens:
            if len(vocab) < 50000:  # Limit vocab size
                vocab[token] = len(vocab)
                
        return vocab
        
    def _tokenize(self, text: str) -> List[int]:
        """Convert text to token ids."""
        tokens = text.strip().split()
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        
    def _pad_sequence(self, sequence: List[int]) -> List[int]:
        """Pad or truncate sequence to max_seq_length."""
        if len(sequence) > self.max_seq_length:
            return sequence[:self.max_seq_length]
        else:
            return sequence + [self.vocab['<pad>']] * (self.max_seq_length - len(sequence))
            
    def __len__(self) -> int:
        return len(self.texts)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        tokens = self._tokenize(text)
        
        if self.is_training:
            # For training, create input and target sequences
            input_seq = [self.vocab['<start>']] + tokens
            target_seq = tokens + [self.vocab['<end>']]
            
            # Pad sequences
            input_seq = self._pad_sequence(input_seq)
            target_seq = self._pad_sequence(target_seq)
            
            return {
                'input_ids': torch.tensor(input_seq),
                'labels': torch.tensor(target_seq)
            }
        else:
            # For evaluation, only create input sequence
            input_seq = [self.vocab['<start>']] + tokens
            input_seq = self._pad_sequence(input_seq)
            
            return {
                'input_ids': torch.tensor(input_seq)
            }

def get_dataloader(
    data_path: str,
    batch_size: int,
    max_seq_length: int,
    tokenizer_path: Optional[str] = None,
    is_training: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create a DataLoader for the dataset."""
    dataset = TextDataset(
        data_path,
        max_seq_length,
        tokenizer_path,
        is_training
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True
    ) 