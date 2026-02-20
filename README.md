# RLQuant: Reinforcement Learning for Options Hedging

A reinforcement learning framework that learns optimal hedging strategies for European vanilla options. This project compares learned hedging strategies against the theoretical Black-Scholes model.

## Overview

**RLQuant** uses actor-critic reinforcement learning to solve the options hedging problem:
- **Actor**: Learns the optimal delta (hedging position) as a function of market state
- **Critic**: Learns the option value as a function of market state
- **Environment**: Simulates geometric Brownian motion stock price paths with vanilla options

The key innovation is using pretraining with domain knowledge:
1. **Actor Pretraining**: Initialized with Black-Scholes delta hedging
2. **Critic Pretraining**: Initialized with Monte Carlo option valuation
3. **Fine-tuning**: Actor-critic learning via experience replay

## Project Structure

```
RLQuant/
├── Envs.py              # Trading environment (Black-Scholes process, VanillaEnv)
├── model.py             # Neural network architectures (actor, critic)
├── pretrain.py          # Pretraining pipeline using domain knowledge
├── main.py              # Main training loop with actor-critic learning
├── blackscholes.py      # Black-Scholes pricing and Greeks calculations
├── replay_buffer.py     # Experience replay buffer for training
├── without_pretrain.py  # Baseline: training without pretraining
└── demo.ipynb          # Demonstration notebook
```

## Key Components

### Environment (`Envs.py`)
- **BlackProcess**: Generates stock price paths using geometric Brownian motion
  - Parameters: initial price (S0), drift (r), volatility (sigma), tenor (days)
- **VanillaEnv**: Options trading environment
  - Observation: `(moneyness, moneyness², time_to_maturity, time_to_maturity², moneyness × time_to_maturity)`
  - Action: Delta hedging position (-1 to 1)
  - Reward: P/L from hedging

### Models (`model.py`)
- **Actor**: `Dense(64) → Dense(64) → Dense(1, tanh)`
  - Input: Market observation (5D)
  - Output: Delta position (-1 to 1)
- **Critic**: `Dense(64) → Dense(64) → Dense(1, sigmoid)`
  - Input: Market observation (5D)
  - Output: Option value (0 to 1)

### Pretraining (`pretrain.py`)
**Actor Pretraining**:
- Collects episodes with fixed action (0.5)
- Trains network to maximize cumulative P/L from delta hedging
- Loss: MSE between network output and optimal Black-Scholes delta

**Critic Pretraining**:
- Uses pretrained actor to generate episode data
- Computes target option values via backward induction (Monte Carlo style)
- Loss: MSE between predicted and target values

### Training (`main.py`)
**Actor-Critic Algorithm**:
- Critic Loss: `(V(S') × df - R + V(S) - δ × ΔS)²`
- Actor Loss: `(δ × (ΔS - μ) + V(S') × df - V(S))²`
- Soft target update: `target ← target × (1-τ) + critic × τ`
- Training: 50 episodes, batch size 32, replay buffer 1024

## Installation

### Requirements
- Python 3.7+
- TensorFlow 2.x
- NumPy

### Setup
```bash
# Install dependencies
pip install tensorflow numpy

# Run training
python main.py
```

## Usage

### Training the Model
```bash
python main.py
```

Output shows:
- Pretraining progress (actor and critic)
- Training episodes with hedge P/L and option payoff
- Final comparison: RL model vs Black-Scholes

Example output:
```
pretrain actor
pretrain critic
train like actor-critic
episode 0
total hedge P/L: 0.0234, option payoff: 0.0500
...
episode 49
total hedge P/L: 0.0198, option payoff: 0.0450

by RL model:
 option value: 0.0234, delta: 0.4567
by black-scholes model:
 option value: 0.0231, delta: 0.4523
```

### Pretraining Only
```bash
python pretrain.py
```

Evaluates pretraining performance and shows learned option values vs target values.

### Demo
See `demo.ipynb` for interactive examples and visualizations.

## How It Works

### Problem Formulation
For a European vanilla option with strike K and expiry T:
- At each time step t, we choose a hedging position δ(t)
- Stock price moves by ΔS, generating P/L: δ × ΔS
- Discount factor: df = e^(-r/365)
- Goal: Learn δ(t) to minimize hedging error

### Learning Process
1. **Pretraining Phase**:
   - Initialize actor with Black-Scholes behavior
   - Initialize critic with Monte Carlo values
   - Provides warm start with financial domain knowledge

2. **Fine-tuning Phase**:
   - Collect experiences in replay buffer
   - Update critic: predict option value V(S)
   - Update actor: maximize hedge effectiveness
   - Soft update target network for stability

### Convergence
The learned delta converges toward the theoretical Black-Scholes delta, demonstrating that RL can discover optimal hedging strategies from data.

## Configuration

Main parameters in `main.py`:
```python
S0, r, vol, days, strike = 1, 0.01, 0.3, 30, 1.1  # Market parameters
n_samples = 2**12                                  # Pretraining samples
n_hidden = [64, 64]                              # Hidden layer sizes
n_episodes = 50                                   # Training episodes
batch_size = 32                                   # Mini-batch size
tau = 0.1                                         # Soft update coefficient
```

## Performance Metrics

- **Option Value**: Learned value vs Black-Scholes price
- **Delta**: Learned hedge position vs theoretical delta
- **Hedge P/L**: Cumulative profit/loss from hedging

## References

- Black-Scholes formula for European options
- Actor-Critic methods with target networks
- Experience replay for stability
- Geometric Brownian Motion for stock prices
