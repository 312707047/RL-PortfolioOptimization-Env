# Portfolio Optimization Environment

This project implements a portfolio optimization environment using OpenAI Gym (now Gymnasium) for reinforcement learning applications in financial trading.

## Overview

The environment simulates a portfolio management scenario where an agent can allocate capital across multiple assets. It includes features such as:

- Multi-asset trading
- Transaction costs and taxes
- Customizable lookback window for historical data
- Flexible state representation

## Key Components

1. `AssetManager`: Handles the core logic of portfolio management, including:
   - Asset valuation
   - Order execution
   - State management

2. `PortfolioOptEnv_gym`: The main Gym environment class, which:
   - Implements the Gym interface (reset, step, etc.)
   - Calculates rewards and episode metrics
   - Handles action mapping and state generation

3. Utility functions for creating observation spaces

## Usage

To use this environment in your reinforcement learning experiments:

1. Initialize the environment with appropriate parameters
2. Use the standard Gym interface for interacting with the environment

Example usage of the environment with `stable baselines3`:
```python
def train(args):
    cwd = f"./experiments/{args.agent}/{args.dataset_name}_{args.version}"

    train_data = pd.read_parquet(train_dataset_path)
    
    env_args = {
        "env_name": args.env_name,
        "num_envs": args.num_envs,
        "max_step": args.max_step,
        "status": "train",
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "if_discrete": False,
        "lookback_window": lookback,
        "break_step": args.break_step,
        "total_value": args.total_value,
        "commision": args.transaction_cost,
        "tax_rate": args.tax_rate,
        "device": "cuda",
        "data": train_data,
        "cwd": cwd,
    }
    
    train_args = env_args.copy()

    train_env = PortfolioOptEnv_gym(**train_args)
    train_env = SubprocVecEnv([lambda: train_env for _ in range(args.num_envs)])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=args.random_seed,
    )
    
    model.learn(total_timesteps=args.break_step)
    model.save(f"{cwd}/model/last_model")
```
### Data format
Data should contains the following columns. Columns after `ticker` should be your features sent to agents.

![alt text](./img/image.png)