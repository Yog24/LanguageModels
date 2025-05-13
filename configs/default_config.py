"""Default configuration for language models."""

from typing import Dict, Any

def get_default_config() -> Dict[str, Any]:
    return {
        # Model architecture
        'model_type': 'gpt',  # ['gpt', 'transformer', 'rnn', 'lstm']
        
        # Model dimensions
        'd_model': 256,  # Embedding dimension
        'd_ff': 1024,  # Feed-forward dimension
        'n_heads': 8,  # Number of attention heads
        'n_layers': 6,  # Number of layers
        'hidden_size': 256,  # Hidden size for RNN/LSTM
        'max_seq_length': 512,  # Maximum sequence length
        'vocab_size': 50257,  # GPT-2 vocabulary size
        
        # Dropout and regularization
        'dropout': 0.1,
        'weight_decay': 0.01,
        
        # Training
        'batch_size': 32,
        'learning_rate': 3e-4,
        'warmup_steps': 1000,
        'max_steps': 100000,
        'eval_steps': 1000,
        'save_steps': 1000,
        
        # Generation
        'max_length': 100,
        'temperature': 1.0,
        'top_k': 50,
        'top_p': 0.9,
        
        # Special tokens
        'pad_token': 0,
        'start_token': 1,
        'end_token': 2,
        'unk_token': 3,
        
        # RNN/LSTM specific
        'bidirectional': False,  # Only for LSTM
        
        # Optimizer
        'optimizer': 'adamw',  # ['adam', 'adamw', 'sgd']
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'clip_grad_norm': 1.0,
        
        # Data
        'train_data_path': 'data/gigaspeech/train.txt',
        'eval_data_path': 'data/gigaspeech/eval.txt',
        'tokenizer_path': 'tokenizer.json',
        
        # Logging and saving
        'log_dir': 'logs',
        'save_dir': 'checkpoints',
        'experiment_name': 'default',
        'use_wandb': False,
        'wandb_project': 'language-models',
    }

def get_small_config() -> Dict[str, Any]:
    """Configuration for small models (for testing)."""
    config = get_default_config()
    config.update({
        'd_model': 128,
        'd_ff': 512,
        'n_heads': 4,
        'n_layers': 3,
        'hidden_size': 128,
        'max_seq_length': 256,
        'batch_size': 16,
    })
    return config

def get_large_config() -> Dict[str, Any]:
    """Configuration for large models."""
    config = get_default_config()
    config.update({
        'd_model': 512,
        'd_ff': 2048,
        'n_heads': 16,
        'n_layers': 12,
        'hidden_size': 512,
        'max_seq_length': 1024,
        'batch_size': 8,
    })
    return config

def get_model_specific_config(model_type: str) -> Dict[str, Any]:
    """Get model-specific configuration overrides."""
    config = get_default_config()
    
    if model_type == 'transformer':
        config.update({
            'n_heads': 8,
            'd_model': 512,
        })
    elif model_type == 'rnn':
        config.update({
            'hidden_size': 512,
            'n_layers': 3,
        })
    elif model_type == 'lstm':
        config.update({
            'hidden_size': 512,
            'n_layers': 3,
            'bidirectional': True,
        })
        
    return config 