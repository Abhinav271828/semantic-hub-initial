# Semantic Hub: Multi-Datatype Language Model Training & Analysis

A research codebase for training GPT-style language models on synthetic datasets with multiple datatypes (partitions) and analyzing their hidden representations for potential collapse or separation across these datatypes.

## Overview

This project investigates how language models organize and represent semantically equivalent but syntactically different data partitions. The codebase supports:

- Training transformer models on synthetic grammar-based and arithmetic datasets
- Creating datasets with multiple "datatypes" (e.g., different vocabulary tokens representing the same semantic concepts)
- Analyzing learned representations to detect if datatypes collapse or remain separated
- Running intervention experiments to understand datatype representations

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Training](#training)
- [Analysis Methods](#analysis-methods)
- [Configuration](#configuration)
- [Results](#results)

## Installation

```bash
# Create virtual environment
python -m venv sem
source sem/bin/activate  # On Windows: sem\Scripts\activate

# Install dependencies
pip install torch numpy matplotlib tqdm nltk hydra-core wandb pyyaml omegaconf
```

## Quick Start

```bash
# Train a model on arithmetic dataset
python train.py data.language=arith

# Train on expression grammar
python train.py data.language=expr

# Run analysis on a trained model
python analysis/analysis.py --run_name YOUR_RUN_ID --all

# Track representation evolution across checkpoints
python analysis/trends.py --run_name YOUR_RUN_ID
```

## Project Structure

```
.
├── train.py                    # Main training script
├── config/
│   └── conf.yaml              # Hydra configuration file
├── dgp/                       # Data Generation Process
│   ├── __init__.py
│   ├── dataloader.py          # Dataset classes and dataloader
│   ├── PCFG.py                # Probabilistic Context-Free Grammar implementation
│   ├── arith.py               # Arithmetic dataset (addition expressions)
│   └── utils.py               # Prior distributions for grammar rules
├── model/
│   └── model.py               # GPT architecture implementation
├── evals/
│   ├── __init__.py
│   └── evals.py               # Grammaticality and validity evaluation
├── analysis/
│   ├── __init__.py
│   ├── analysis.py            # Core analysis functions
│   └── trends.py              # Cross-checkpoint evolution analysis
├── utils/
│   ├── __init__.py
│   ├── logging.py             # Logging, W&B integration, checkpointing
│   └── optimizer.py           # Optimizer configuration and LR scheduling
├── results/                   # Training outputs (models, configs, grammars)
├── logs/                      # Training logs
└── wandb/                     # Weights & Biases tracking
```

## Datasets

The codebase supports four types of synthetic datasets, each with configurable multiple datatypes:

### 1. Arithmetic Dataset (`arith`)

**Location**: `dgp/arith.py`

Generates basic addition expressions in two datatypes:
- **Datatype 0**: Numeric notation (e.g., `1 2 3 + 4 5 6 = 5 7 9`)
- **Datatype 1**: Word notation (e.g., `one hundred twenty three + four hundred fifty six = five hundred seventy nine`)

**Features**:
- Supports numbers 0-1998
- Validates arithmetic correctness during evaluation
- Vocabulary includes digits, number words, operators (`+`, `=`), and special tokens

**Configuration**:
```yaml
data:
  language: 'arith'
  D: 2  # Number of datatypes
  T: 1  # Temperature for datatype distribution
```

### 2. Expression Grammar (`expr`)

**Location**: `dgp/PCFG.py`

Generates mathematical expressions using prefix or postfix notation with multiple datatypes.

**Grammar Rules**:
```
S -> Expr
Expr -> OpExpr | Digit
OpExpr -> UnOp Expr | BinOp Expr Expr | TernOp Expr Expr Expr  # prefix
```

**Configuration**:
```yaml
data:
  language: 'expr'
  config:
    n_digits: 10      # Number of digit symbols
    n_ops: 3          # Number of operator symbols
    postfix: False    # Use prefix (True for postfix)
```

### 3. Dyck Languages (`dyck`)

**Location**: `dgp/PCFG.py`

Generates balanced bracket sequences with multiple bracket types.

**Grammar Rules**:
```
S -> S S | Brack_0 | Brack_1 | ...
Brack_i -> 'o_i' S 'c_i' | 'o_i' 'c_i'
```

**Configuration**:
```yaml
data:
  language: 'dyck'
  config:
    n_brackets: 2  # Number of bracket types
```

### 4. English-like Grammar (`english`)

**Location**: `dgp/PCFG.py`

Generates simple English-like sentences.

**Grammar Rules**:
```
S -> NP VP
NP -> Adj N | NP Conj NP | Pro
VP -> V | V NP | VP Adv | VP Conj VP
```

**Configuration**:
```yaml
data:
  language: 'english'
  config:
    n_nouns: 10
    n_verbs: 10
    n_adjectives: 10
    n_pronouns: 10
    n_adverbs: 10
    n_conjunctions: 2
```

### Datatype System

All datasets support multiple datatypes through vocabulary replication:
- **D**: Number of distinct datatypes (e.g., D=2 for numeric vs. word representations)
- **T**: Controls datatype frequency distribution via softmax([0, -T, -2T, ..., -(D-1)T])
- Each token is duplicated D times with suffix `-0`, `-1`, ..., `-(D-1)`
- Example: `dig5` becomes `dig5-0` (datatype 0) and `dig5-1` (datatype 1)

## Training

### Main Training Script

**File**: `train.py`

Coordinates the entire training process:

1. **Initialization**
   - Sets up Weights & Biases logging
   - Seeds random number generators
   - Saves configuration

2. **Data Loading**
   - Creates dataloader with specified language and configuration
   - Configures datatype distribution based on D and T parameters

3. **Model Setup**
   - Initializes GPT model with vocabulary size from dataset
   - Moves model to device (CPU/CUDA/MPS)
   - Optionally compiles model with PyTorch 2.0

4. **Training Loop**
   - Mixed precision training (bfloat16)
   - Cosine learning rate schedule with warmup
   - Gradient clipping
   - Periodic evaluation and checkpointing

5. **Evaluation** (periodic during training)
   - Grammar/validity checking on generated samples
   - Log probability analysis
   - Parse tree depth statistics (for PCFG datasets)

### Key Training Features

- **Mixed Precision**: Uses bfloat16 for efficient training
- **Flash Attention**: Leverages `scaled_dot_product_attention` for speed
- **Gradient Clipping**: Prevents exploding gradients
- **Weight Tying**: Shares embedding and output projection weights
- **Cosine LR Schedule**: Smooth learning rate decay with linear warmup

### Model Architecture

**File**: `model/model.py`

GPT-style decoder-only transformer:

**Components**:
- **Token Embeddings**: Maps vocabulary to embedding space
- **Position Embeddings**: Learned positional encodings
- **Transformer Blocks**: Self-attention + MLP layers
  - Multi-head causal self-attention
  - Layer normalization
  - Optional MLP (can be disabled)
  - Residual connections
- **Language Modeling Head**: Projects to vocabulary logits

**Features**:
- Configurable layers, heads, embedding dimension
- Supports attention map extraction for analysis
- Multiple sampling strategies (greedy, stochastic, top-k/top-p)
- Log-likelihood computation for sequences

## Analysis Methods

### Core Analysis Functions

**File**: `analysis/analysis.py`

#### 1. Visualization (`visualize_outputs_with_logits`)

Generates samples and displays per-token logits as heatmaps.

**Features**:
- Shows vocabulary (y-axis) vs. generated tokens (x-axis)
- Highlights chosen tokens with borders
- Useful for understanding model confidence

**Usage**:
```python
visualize_outputs_with_logits(model, grammar, cfg, num_samples=5)
```

#### 2. Distance Analysis

Two variants depending on unit of comparison:

##### Token-Level (`analyze_datatype_embedding_distances_tokens`)

Analyzes embedding distances between tokens of different datatypes at layer 1.

**Method**:
1. Generate same base sequence in both datatypes
2. Pass through model separately
3. Compute cosine similarity matrix between corresponding tokens
4. Compute average pairwise similarity within sequences as baseline

**Outputs**:
- Average cosine similarity within sequences (baseline)
- Average cosine similarity between datatypes
- Similarity heatmap across vocabulary
- Average embedding norm

**Usage**:
```python
results = analyze_datatype_embedding_distances_tokens(
    model, dataloader, grammar,
    num_sequences=1000,
    showplot=True
)
```

##### Sequence-Level (`analyze_datatype_embedding_distances_sequences`)

Uses EOS token embeddings to compare entire sequences.

**Method**:
1. Generate parallel sequences (same semantic content, different datatypes)
2. Extract EOS token representations from layer 1
3. Compute cross-datatype cosine similarities

**Outputs**:
- Average similarity within each datatype
- Average similarity across datatypes

#### 3. Linear Regression Analysis

Two variants for token-level and sequence-level:

##### Token-Level (`linear_regression_datatype_separation_tokens`)

Trains a linear classifier to predict datatype from token embeddings.

**Method**:
1. Collect layer 1 embeddings with datatype labels
2. Split into train/test (80/20)
3. Fit linear regression with least squares
4. Evaluate with accuracy, precision, recall

**Visualization**:
- 2D PCA projection of embeddings
- Decision boundary visualization
- Color-coded by datatype

**Usage**:
```python
results = linear_regression_datatype_separation_tokens(
    model, dataloader, grammar,
    max_sequences=1200,
    showplot=True
)
```

##### Sequence-Level (`linear_regression_datatype_separation_sequences`)

Same as token-level but uses EOS embeddings to classify entire sequences.

#### 4. Representation Intervention Experiment

**Function**: `representation_intervention_experiment`

Tests if swapping datatype representations affects generation.

**Method**:
1. Generate same sequence in both datatypes
2. Pass prefix through model up to swap position
3. Capture representation from datatype 0
4. Replace representation in datatype 0 with captured value
5. Continue generation and check validity

**Categories of Results**:
- **(i) Empty continuations**: Model stops immediately
- **(ii) Invalid nonempty continuations**: Generates invalid sequences
- **(iii) Valid nonempty continuations**: Generates valid sequences

**Usage**:
```python
results = representation_intervention_experiment(
    model, grammar,
    num_experiments=100,
    verbose=True
)
```

### Evolution Analysis

**File**: `analysis/trends.py`

Analyzes how representations evolve across training checkpoints.

**Features**:
- Automatically detects all checkpoints in run directory
- Runs all analysis methods for each checkpoint
- Tracks metrics over training:
  - Normalized distances between datatypes
  - Overall average embedding distances
  - Linear regression accuracy
  - Intervention experiment percentages

**Outputs**:
- Figure 1: Normalized distances (relative to baseline)
- Figure 2: Raw distances with baseline overlay
- Figure 3: Classification accuracy over time
- Figure 4: Intervention results percentages

**Usage**:
```bash
python analysis/trends.py --run_name YOUR_RUN_ID --num_sequences 1000
```

## Evaluation

**File**: `evals/evals.py`

### Grammar Evaluation (`grammar_evals`)

For PCFG-based datasets (expr, dyck, english):

**Checks**:
- Grammaticality using Viterbi parser
- Parse tree depth
- Log probabilities from grammar vs. model

**Outputs**:
- Validity percentage
- Grammar log probabilities (max, min, mean)
- Model log probabilities
- Sequence lengths
- Parse tree depths

### Arithmetic Evaluation (`arith_evals`)

For arithmetic dataset:

**Checks**:
- Arithmetic correctness (a + b = c)
- Format validation

**Outputs**:
- Validity percentage
- Model log probabilities

## Configuration

**File**: `config/conf.yaml`

Uses Hydra for configuration management.

### Key Parameters

```yaml
# Deployment
deploy: True          # Enable W&B logging and checkpointing
tag: scratch          # Experiment tag for organization
seed: 42              # Random seed

# Device
device: "mps"         # "cuda", "cpu", or "mps"
bf16: True            # Use bfloat16 mixed precision
epochs: 5             # Number of training epochs

# Data
data:
  language: 'expr'    # Dataset type
  unit: 'tok'         # Comparison unit: 'tok' or 'seq'
  config:             # Language-specific config
    n_digits: 10
    n_ops: 3
  D: 2                # Number of datatypes
  T: 1                # Datatype distribution temperature
  alpha: 1e5          # Prior concentration parameter
  prior_type: 'dirichlet'  # Prior distribution type
  max_sample_length: 128   # Maximum sequence length
  num_iters: 1e4      # Iterations per epoch
  batch_size: 128

# Model
model:
  compile: False      # Use torch.compile (PyTorch 2.0+)
  context_size: 256   # Maximum context length
  n_layer: 2          # Number of transformer layers
  n_head: 2           # Number of attention heads
  n_embd: 128         # Embedding dimension
  dropout: 0.0        # Dropout probability
  bias: False         # Use bias in linear layers
  mlp: True           # Include MLP in transformer blocks

# Optimizer
optimizer:
  learning_rate: 1e-3
  weight_decay: 1e-4
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0      # Gradient clipping threshold
  decay_lr: True      # Use cosine decay
  warmup_iters: 200   # Warmup steps
  min_lr: 9e-4        # Minimum learning rate

# Evaluation
eval:
  grammar: True       # Run grammar/validity checks

# Logging
log:
  save_multiple: True      # Save multiple checkpoints
  log_interval: 1000       # Steps between train logs
  eval_interval: 1000      # Steps between evaluations
  save_interval: 100       # Steps between checkpoints
  print_gen_samples: 0     # Number of samples to print
```

### Hydra Override

Override any parameter from command line:

```bash
# Change dataset
python train.py data.language=dyck

# Adjust model size
python train.py model.n_layer=4 model.n_head=4 model.n_embd=256

# Change number of datatypes
python train.py data.D=3 data.T=2

# Multiple overrides
python train.py data.language=arith model.n_layer=6 epochs=10
```

## Results

Training outputs are organized by experiment:

```
results/
└── scratch/              # Based on 'tag' in config
    └── {wandb_run_id}/   # Unique run identifier
        ├── conf.yaml     # Saved configuration
        ├── grammar/
        │   └── PCFG.pkl  # Saved grammar/dataset object
        ├── ckpt_0.pt     # Initial checkpoint
        ├── ckpt_100.pt   # Checkpoint at step 100
        └── ...           # More checkpoints
```

### Checkpoint Contents

Each checkpoint (.pt file) contains:
- `net`: Model state dict
- `optimizer`: Optimizer state dict
- `iter`: Training iteration
- `config`: Full configuration object

### Loading Checkpoints

```python
from analysis.analysis import load_model_and_grammar

model, grammar, cfg, dataloader = load_model_and_grammar(
    run_name="YOUR_RUN_ID",
    ckpt_name="ckpt_100001.pt"
)
```

## Utilities

### Logging (`utils/logging.py`)

**Functions**:
- `init_wandb()`: Initialize Weights & Biases
- `save_config()`: Save configuration to YAML
- `log_train()`: Log training metrics
- `log_eval()`: Log evaluation metrics
- `save_model()`: Save model checkpoint
- `set_seed()`: Set random seeds for reproducibility

### Optimizer (`utils/optimizer.py`)

**Functions**:
- `configure_optimizers()`: Set up AdamW with weight decay groups
- `update_cosine_warmup_lr()`: Update learning rate with cosine schedule
- `move_to_device()`: Move tensors to specified device

## Common Use Cases

### 1. Train a model and analyze representations

```bash
# Train
python train.py data.language=expr data.D=2 epochs=10

# After training, note the wandb run_id (e.g., "abc123xyz")

# Analyze final checkpoint
python analysis/analysis.py --run_name abc123xyz --all

# Track evolution across training
python analysis/trends.py --run_name abc123xyz --num_sequences 1000
```

### 2. Compare different datatype configurations

```bash
# 2 datatypes with equal probability
python train.py data.D=2 data.T=0

# 2 datatypes with unequal probability
python train.py data.D=2 data.T=2

# 4 datatypes
python train.py data.D=4 data.T=1
```

### 3. Run specific analyses

```bash
# Only distance analysis
python analysis/analysis.py --run_name abc123xyz --distance

# Only linear regression
python analysis/analysis.py --run_name abc123xyz --linear_regression

# Only intervention experiments
python analysis/analysis.py --run_name abc123xyz --intervention

# Customize number of samples
python analysis/analysis.py --run_name abc123xyz --all --num_sequences 5000
```

### 4. Test on arithmetic reasoning

```bash
# Train arithmetic model
python train.py data.language=arith data.config.n_digits=10 epochs=10

# Analyze how well it learns addition
python analysis/analysis.py --run_name YOUR_RUN_ID --all
```

## Research Questions

This codebase is designed to investigate:

1. **Representation Collapse**: Do models collapse representations of semantically equivalent tokens from different datatypes?

2. **Linear Separability**: Can datatypes be linearly separated in the representation space?

3. **Evolution Dynamics**: How do representations evolve during training? Do they start separated and converge, or vice versa?

4. **Causal Role**: Do datatype-specific representations causally influence generation, or are they inert?

5. **Architectural Effects**: How do model size, depth, and attention heads affect representation structure?

6. **Data Distribution Effects**: How does the relative frequency of datatypes (controlled by T) affect learning?

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{semantic_hub,
  title = {Semantic Hub: Multi-Datatype Language Model Training and Analysis},
  year = {2024},
  author = {[Your Name]},
  url = {[Your Repository URL]}
}
```

## License

[Specify your license here]

## Contact

[Your contact information]

## Acknowledgments

- GPT implementation inspired by Andrej Karpathy's nanoGPT
- PCFG implementation uses NLTK for parsing
- Training infrastructure uses Weights & Biases for experiment tracking
