
import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.config.settings import get_settings
from src.models.actor_critic import HybridActorCritic
from src.agent.ppo_agent import PPOAgent
from src.environment.trading_env import TradingEnvironment
from src.data.csv_loader import CSVDataLoader
from src.utils.checkpoint import CheckpointManager

def run_backtest():
    print("STDOUT: Starting v2 Backtest Script")
    sys.stdout.flush()
    
    settings = get_settings()
    # Force CPU to avoid CUDA hangs
    settings.device = "cpu"
    
    csv_name = "XAGUSD_FEB_2026.csv"
    csv_path = settings.historical_data_dir / csv_name
    checkpoint_path = settings.checkpoint_dir / "best_model.pt"
    
    print(f"STDOUT: Loading data from {csv_path}")
    sys.stdout.flush()
    
    loader = CSVDataLoader(csv_path)
    loader.load()
    candles = loader.candles
    print(f"STDOUT: Loaded {len(candles)} candles")
    sys.stdout.flush()
    
    print(f"STDOUT: Initializing model (input_dim={settings.model_input_dim})")
    sys.stdout.flush()
    
    model = HybridActorCritic(
        sequence_length=settings.model_sequence_length,
        input_dim=settings.model_input_dim,
        hidden_dim=settings.model_hidden_size,
        num_layers=settings.model_num_layers,
        num_heads=settings.model_attention_heads,
        prediction_dim=settings.model_input_dim,
        device=settings.device,
    )
    
    agent = PPOAgent(model)
    checkpoint_mgr = CheckpointManager(settings.checkpoint_dir)
    checkpoint_mgr.load(model, checkpoint_path=checkpoint_path)
    print("STDOUT: Checkpoint loaded")
    sys.stdout.flush()
    
    from src.environment.reward_calculator import AdaptiveRewardCalculator

    env = TradingEnvironment(
        sequence_length=settings.model_sequence_length,
        lot_size=settings.trading_lot_size,
        input_dim=settings.model_input_dim,
        reward_calculator=AdaptiveRewardCalculator(),
    )
    
    initial_candles = candles[:settings.model_sequence_length]
    obs, _ = env.reset(options={"initial_candles": initial_candles})
    agent.reset_hidden()
    
    print("STDOUT: Starting loop...")
    sys.stdout.flush()
    
    total_reward = 0.0
    steps = 0
    last_print = time.time()
    
    for i, candle in enumerate(candles[settings.model_sequence_length:]):
        env.receive_candle(candle)
        action, info = agent.get_action(obs, deterministic=True)
        env_action = {
            "prediction": info["prediction"],
            "trading_action": info["trading_action"],
        }
        obs, reward, terminated, truncated, step_info = env.step(env_action)
        total_reward += reward
        steps += 1
        
        if time.time() - last_print > 5:
            print(f"STDOUT: Progress: {i}/{len(candles)-120} steps | Total PnL: {env.position_manager.get_total_pnl():.2f}")
            sys.stdout.flush()
            last_print = time.time()
            
        if terminated:
            print("STDOUT: Episode terminated early (Max Loss)")
            sys.stdout.flush()
            break
            
    stats = env.get_statistics()
    print("--- BACKTEST RESULTS ---")
    print(f"Steps: {steps}")
    print(f"Total PnL: {stats['trading_stats']['total_pnl']:.2f}")
    print(f"Win Rate: {stats['trading_stats']['win_rate']:.2%}")
    print(f"Total Trades: {stats['trading_stats']['total_trades']}")
    print("------------------------")
    sys.stdout.flush()

if __name__ == "__main__":
    import time
    try:
        run_backtest()
    except Exception as e:
        print(f"STDOUT: Error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
