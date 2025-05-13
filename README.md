# Language Model Experiments

This repository contains implementations of various language models and an experiment framework for comparing and analyzing their performance.

## Models

The following models are implemented:

- GPT (Generative Pre-trained Transformer)
- Transformer (Encoder-Decoder architecture)
- RNN (Recurrent Neural Network)
- LSTM (Long Short-Term Memory)

## Project Structure

```
.
├── configs/
│   └── default_config.py     # Configuration settings
├── data/
│   ├── train.txt            # Training data
│   └── eval.txt             # Evaluation data
├── src/
│   ├── models/
│   │   ├── base_model.py    # Base model class
│   │   ├── gpt_model.py     # GPT implementation
│   │   ├── transformer_model.py  # Transformer implementation
│   │   ├── rnn_model.py     # RNN implementation
│   │   └── lstm_model.py    # LSTM implementation
│   ├── experiments/
│   │   ├── experiment.py    # Experiment runner
│   │   └── run_experiments.py  # Main script
│   └── utils/
│       ├── data_utils.py    # Data loading utilities
│       └── metrics.py       # Evaluation metrics
├── tests/                   # Test files
├── logs/                    # Training logs
├── checkpoints/             # Model checkpoints
├── requirements.txt         # Dependencies
└── README.md
```

## Features

- Multiple model architectures with consistent interface
- Configurable hyperparameters
- Experiment framework for model comparison
- Ablation studies
- Metrics tracking (loss, perplexity, accuracy, BLEU)
- Integration with Weights & Biases for experiment tracking
- Checkpoint saving and loading
- Data preprocessing and batching

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Single Model Training

```bash
python -m src.experiments.run_experiments --mode single --model_type gpt --config_size default
```

### Model Comparison

```bash
python -m src.experiments.run_experiments --mode comparison
```

### Ablation Study

```bash
python -m src.experiments.run_experiments --mode ablation --model_type gpt
```

## Configuration

The default configuration can be found in `configs/default_config.py`. You can modify the hyperparameters there or create custom configurations.

Available configuration sizes:
- `small`: For quick testing
- `default`: Standard training
- `large`: For better performance (requires more compute)

## Ablation Studies

The framework supports ablation studies on various hyperparameters:

- Learning rate
- Number of layers
- Dropout rate
- Model-specific parameters (e.g., number of attention heads, hidden size)

Results are saved in JSON format for further analysis.

## Metrics

The following metrics are computed during training and evaluation:

- Loss
- Perplexity
- Token-level accuracy
- BLEU score (for generation tasks)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

