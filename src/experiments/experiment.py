import os
import json
import torch
import wandb
import numpy as np
from typing import Dict, Any, List, Optional
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models.base_model import BaseModel
from ..utils.data_utils import get_dataloader
from ..utils.metrics import compute_metrics

class Experiment:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = BaseModel.create_model(config['model_type'], config)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize logging
        if config['use_wandb']:
            wandb.init(project=config['wandb_project'], config=config)
            
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(
            os.path.join(config['log_dir'], config['experiment_name'])
        )
            
        # Create directories
        os.makedirs(config['log_dir'], exist_ok=True)
        os.makedirs(config['save_dir'], exist_ok=True)
        
    def _create_optimizer(self):
        if self.config['optimizer'] == 'adam':
            return Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                betas=(self.config['beta1'], self.config['beta2']),
                eps=self.config['epsilon'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'adamw':
            return AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                betas=(self.config['beta1'], self.config['beta2']),
                eps=self.config['epsilon'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            return SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
            
    def _create_scheduler(self):
        def lr_lambda(step):
            warmup_steps = self.config['warmup_steps']
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
            
        return LambdaLR(self.optimizer, lr_lambda)
        
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """Train the model."""
        self.model.train()
        global_step = 0
        best_eval_loss = float('inf')
        epoch = 0
        
        progress_bar = tqdm(total=self.config['max_steps'], desc='Training')
        
        while global_step < self.config['max_steps']:
            epoch += 1
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_dataloader:
                if global_step >= self.config['max_steps']:
                    break
                    
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch['input_ids'])
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Update epoch statistics
                epoch_loss += loss.item()
                num_batches += 1
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['clip_grad_norm']
                )
                
                # Update parameters
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Logging
                if self.config['use_wandb']:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/global_step': global_step,
                    })
                
                # TensorBoard logging for each step
                self.writer.add_scalar('Loss/train_step', loss.item(), global_step)
                self.writer.add_scalar('Learning_rate', self.scheduler.get_last_lr()[0], global_step)
                
                # Evaluation
                if eval_dataloader is not None and global_step % self.config['eval_steps'] == 0:
                    eval_results = self.evaluate(eval_dataloader)
                    
                    # TensorBoard logging for evaluation
                    self.writer.add_scalar('Loss/eval', eval_results['loss'], global_step)
                    self.writer.add_scalar('Perplexity/eval', eval_results['perplexity'], global_step)
                    
                    if eval_results['loss'] < best_eval_loss:
                        best_eval_loss = eval_results['loss']
                        self.save_checkpoint(f"best_model.pt")
                        
                    if self.config['use_wandb']:
                        wandb.log({
                            'eval/loss': eval_results['loss'],
                            'eval/perplexity': eval_results['perplexity'],
                            'eval/global_step': global_step,
                        })
                
                # Save checkpoint
                if global_step % self.config['save_steps'] == 0:
                    self.save_checkpoint(f"checkpoint_{global_step}.pt")
                
                global_step += 1
                progress_bar.update(1)
            
            # End of epoch logging
            avg_epoch_loss = epoch_loss / num_batches
            self.writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
            
            # Run evaluation at the end of each epoch
            if eval_dataloader is not None:
                eval_results = self.evaluate(eval_dataloader)
                self.writer.add_scalar('Loss/eval_epoch', eval_results['loss'], epoch)
                self.writer.add_scalar('Perplexity/eval_epoch', eval_results['perplexity'], epoch)
                
        progress_bar.close()
        self.writer.close()
        
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch['input_ids'])
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
                total_loss += loss.item()
                total_steps += 1
                
        avg_loss = total_loss / total_steps
        metrics = {
            'loss': avg_loss,
            'perplexity': np.exp(avg_loss)
        }
        
        self.model.train()
        return metrics
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.config['save_dir'], filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }, path)
        
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.config['save_dir'], filename)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.config = checkpoint['config']
        
    @staticmethod
    def run_ablation(
        base_config: Dict[str, Any],
        param_grid: Dict[str, List[Any]],
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        num_epochs: int = 1
    ) -> List[Dict[str, Any]]:
        """Run ablation study over hyperparameters."""
        results = []
        
        # Generate all combinations of parameters
        from itertools import product
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        for params in param_combinations:
            # Create new config with current parameters
            config = base_config.copy()
            config.update(params)
            
            # Initialize experiment
            experiment = Experiment(config)
            
            # Train model
            experiment.train(train_dataloader, eval_dataloader)
            
            # Evaluate model
            eval_results = experiment.evaluate(eval_dataloader)
            
            # Save results
            results.append({
                'params': params,
                'metrics': eval_results
            })
            
            # Log results
            if config['use_wandb']:
                wandb.log({
                    'ablation/params': params,
                    'ablation/loss': eval_results['loss'],
                    'ablation/perplexity': eval_results['perplexity']
                })
                
        return results 