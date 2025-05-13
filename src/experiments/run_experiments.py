"""Script to run experiments and ablation studies."""

import argparse
import json
from typing import Dict, Any

from ..utils.data_utils import get_dataloader
from .experiment import Experiment
from ...configs.default_config import (
    get_default_config,
    get_small_config,
    get_large_config,
    get_model_specific_config
)

def run_single_experiment(config: Dict[str, Any]):
    """Run a single experiment with given configuration."""
    # Get data loaders
    train_dataloader = get_dataloader(
        config['train_data_path'],
        config['batch_size'],
        config['max_seq_length'],
        is_training=True
    )
    
    eval_dataloader = get_dataloader(
        config['eval_data_path'],
        config['batch_size'],
        config['max_seq_length'],
        is_training=False
    )
    
    # Initialize and run experiment
    experiment = Experiment(config)
    experiment.train(train_dataloader, eval_dataloader)
    
    # Final evaluation
    final_metrics = experiment.evaluate(eval_dataloader)
    print(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
    
def run_model_comparison():
    """Compare different model architectures."""
    model_types = ['gpt', 'transformer', 'rnn', 'lstm']
    
    for model_type in model_types:
        print(f"\nRunning experiment with {model_type.upper()} model")
        config = get_model_specific_config(model_type)
        config['model_type'] = model_type
        config['experiment_name'] = f"model_comparison_{model_type}"
        
        run_single_experiment(config)
        
def run_ablation_study(model_type: str):
    """Run ablation studies for a specific model type."""
    base_config = get_model_specific_config(model_type)
    base_config['model_type'] = model_type
    base_config['experiment_name'] = f"ablation_{model_type}"
    
    # Define parameter grid for ablation study
    param_grid = {
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'n_layers': [2, 4, 6],
        'dropout': [0.1, 0.2, 0.3],
    }
    
    if model_type in ['gpt', 'transformer']:
        param_grid.update({
            'n_heads': [4, 8, 16],
            'd_model': [256, 512, 768],
        })
    elif model_type in ['rnn', 'lstm']:
        param_grid.update({
            'hidden_size': [256, 512, 768],
            'bidirectional': [False, True] if model_type == 'lstm' else [False],
        })
    
    # Get data loaders
    train_dataloader = get_dataloader(
        base_config['train_data_path'],
        base_config['batch_size'],
        base_config['max_seq_length'],
        is_training=True
    )
    
    eval_dataloader = get_dataloader(
        base_config['eval_data_path'],
        base_config['batch_size'],
        base_config['max_seq_length'],
        is_training=False
    )
    
    # Run ablation study
    results = Experiment.run_ablation(
        base_config,
        param_grid,
        train_dataloader,
        eval_dataloader
    )
    
    # Save results
    output_file = f"ablation_results_{model_type}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Ablation study results saved to {output_file}")
    
def main():
    parser = argparse.ArgumentParser(description="Run language model experiments")
    parser.add_argument(
        '--mode',
        choices=['single', 'comparison', 'ablation'],
        default='single',
        help='Experiment mode'
    )
    parser.add_argument(
        '--model_type',
        choices=['gpt', 'transformer', 'rnn', 'lstm'],
        default='gpt',
        help='Model type for single experiment or ablation study'
    )
    parser.add_argument(
        '--config_size',
        choices=['default', 'small', 'large'],
        default='default',
        help='Configuration size'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Get appropriate config
        if args.config_size == 'small':
            config = get_small_config()
        elif args.config_size == 'large':
            config = get_large_config()
        else:
            config = get_default_config()
            
        config['model_type'] = args.model_type
        config['experiment_name'] = f"single_{args.model_type}_{args.config_size}"
        
        run_single_experiment(config)
        
    elif args.mode == 'comparison':
        run_model_comparison()
        
    elif args.mode == 'ablation':
        run_ablation_study(args.model_type)
        
if __name__ == "__main__":
    main() 