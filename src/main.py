"""Main entry point for the XAGUSD RL Trader."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import uvicorn

from src.config.settings import get_settings
from src.utils.logging import setup_logging, get_logger


app = typer.Typer(
    name="xagusd-trader",
    help="XAGUSD Deep RL Trading System",
    add_completion=False,
)


@app.command()
def serve(
    host: str = typer.Option(None, "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(None, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers"),
) -> None:
    """Start the API server."""
    settings = get_settings()
    setup_logging(settings.log_level)
    
    host = host or settings.server_host
    port = port or settings.server_port
    
    logger = get_logger(__name__)
    logger.info("Starting server", host=host, port=port)
    
    uvicorn.run(
        "src.server.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level=settings.log_level.lower(),
    )


@app.command()
def pretrain(
    csv_path: Path = typer.Argument(..., help="Path to CSV data file"),
    timesteps: int = typer.Option(1_000_000, "--timesteps", "-t", help="Total timesteps"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", "-c", help="Resume from checkpoint"),
) -> None:
    """Run pre-training on CSV data."""
    settings = get_settings()
    setup_logging(settings.log_level)
    
    logger = get_logger(__name__)
    logger.info("Starting pre-training", csv_path=str(csv_path), timesteps=timesteps)
    
    # Resolve path
    if not csv_path.is_absolute():
        csv_path = settings.historical_data_dir / csv_path
    
    if not csv_path.exists():
        logger.error("CSV file not found", path=str(csv_path))
        raise typer.Exit(1)
    
    # Import here to avoid circular imports
    from src.models.actor_critic import HybridActorCritic
    from src.agent.ppo_agent import PPOAgent, PPOConfig
    from src.environment.reward_calculator import AdaptiveRewardCalculator, RewardCalculator

    # Training config
    train_config = TrainingConfig(
        total_timesteps=timesteps,
        checkpoint_dir=settings.checkpoint_dir,
        tensorboard_dir=settings.tensorboard_dir,
        checkpoint_interval_minutes=settings.training_checkpoint_interval_minutes,
    )

    # Initialize components
    model = HybridActorCritic(
        sequence_length=settings.model_sequence_length,
        input_dim=settings.model_input_dim,
        hidden_dim=settings.model_hidden_size,
        num_layers=settings.model_num_layers,
        num_heads=settings.model_attention_heads,
        dropout=settings.model_dropout,
        prediction_dim=settings.model_input_dim,
        device=settings.device,
    )
    
    config = PPOConfig(
        learning_rate=settings.training_learning_rate,
        gamma=settings.training_gamma,
        gae_lambda=settings.training_gae_lambda,
        clip_epsilon=settings.training_clip_epsilon,
        entropy_coef=settings.training_entropy_coef,
        value_coef=settings.training_value_coef,
        prediction_coef=settings.training_prediction_coef,
        batch_size=settings.training_batch_size,
        num_epochs=settings.training_num_epochs,
    )
    
    agent = PPOAgent(model, config)
    
    # Select reward calculator
    if settings.training_use_adaptive_reward:
        reward_calculator = AdaptiveRewardCalculator()
        logger.info("Using Adaptive Reward Calculator")
    else:
        reward_calculator = RewardCalculator()
        logger.info("Using Standard Reward Calculator")

    env = TradingEnvironment(
        sequence_length=settings.model_sequence_length,
        lot_size=settings.trading_lot_size,
        stop_loss_usd=settings.trading_stop_loss_usd,
        take_profit_usd=settings.trading_take_profit_usd,
        max_loss_usd=settings.trading_max_loss_usd,
        input_dim=settings.model_input_dim,
        reward_calculator=reward_calculator,
    )
    
    # Load checkpoint if provided
    if checkpoint:
        checkpoint_mgr = CheckpointManager(settings.checkpoint_dir)
        checkpoint_mgr.load(model, agent.optimizer, checkpoint_path=checkpoint)
        logger.info("Loaded checkpoint", path=str(checkpoint))
    

    
    trainer = Trainer(agent, env, train_config)
    
    # Load data
    loader = CSVDataLoader(csv_path)
    loader.load()
    logger.info("Loaded CSV data", num_candles=len(loader))
    
    # Train
    state = trainer.train(loader)
    
    logger.info(
        "Training completed",
        timesteps=state.timesteps,
        episodes=state.episodes,
        best_reward=state.best_reward,
    )


@app.command()
def backtest(
    csv_path: Path = typer.Argument(..., help="Path to CSV data file"),
    checkpoint: Path = typer.Option(..., "--checkpoint", "-c", help="Checkpoint to evaluate"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results"),
) -> None:
    """Run backtesting on historical data."""
    settings = get_settings()
    setup_logging(settings.log_level)
    
    logger = get_logger(__name__)
    logger.info("Starting backtest", csv_path=str(csv_path))
    
    # Resolve paths
    if not csv_path.is_absolute():
        csv_path = settings.historical_data_dir / csv_path
    
    if not csv_path.exists():
        logger.error("CSV file not found", path=str(csv_path))
        raise typer.Exit(1)
    
    if not checkpoint.exists():
        logger.error("Checkpoint not found", path=str(checkpoint))
        raise typer.Exit(1)
    
    # Import components
    from src.models.actor_critic import HybridActorCritic
    from src.agent.ppo_agent import PPOAgent
    from src.environment.trading_env import TradingEnvironment
    from src.data.csv_loader import CSVDataLoader
    from src.utils.checkpoint import CheckpointManager
    
    # Initialize
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
    
    env = TradingEnvironment(
        sequence_length=settings.model_sequence_length,
        lot_size=settings.trading_lot_size,
        stop_loss_usd=settings.trading_stop_loss_usd,
        take_profit_usd=settings.trading_take_profit_usd,
        max_loss_usd=settings.trading_max_loss_usd,
        input_dim=settings.model_input_dim,
    )
    
    # Load checkpoint
    checkpoint_mgr = CheckpointManager(settings.checkpoint_dir)
    checkpoint_mgr.load(model, checkpoint_path=checkpoint)
    
    # Load data
    loader = CSVDataLoader(csv_path)
    loader.load()
    
    # Run backtest
    logger.info("Running backtest...")
    
    # Get initial candles
    candles = loader.candles
    initial_candles = candles[:settings.model_sequence_length]
    
    obs, _ = env.reset(options={"initial_candles": initial_candles})
    agent.reset_hidden()
    
    total_reward = 0.0
    steps = 0
    
    for candle in candles[settings.model_sequence_length:]:
        env.receive_candle(candle)
        
        action, info = agent.get_action(obs, deterministic=True)
        
        env_action = {
            "prediction": info["prediction"],
            "trading_action": info["trading_action"],
        }
        
        obs, reward, terminated, truncated, step_info = env.step(env_action)
        
        total_reward += reward
        steps += 1
        
        if terminated:
            break
    
    # Get results
    stats = env.get_statistics()
    trading_stats = stats["trading_stats"]
    
    logger.info(
        "Backtest completed",
        steps=steps,
        total_reward=total_reward,
        total_pnl=trading_stats["total_pnl"],
        win_rate=trading_stats["win_rate"],
        total_trades=trading_stats["total_trades"],
    )
    
    # Save results if output specified
    if output:
        import json
        results = {
            "steps": steps,
            "total_reward": total_reward,
            "trading_stats": trading_stats,
        }
        output.write_text(json.dumps(results, indent=2))
        logger.info("Results saved", path=str(output))


@app.command()
def info() -> None:
    """Show system information."""
    import torch
    
    settings = get_settings()
    
    typer.echo("XAGUSD RL Trader System Info")
    typer.echo("=" * 40)
    typer.echo(f"Environment: {settings.app_env}")
    typer.echo(f"Python: {__import__('sys').version}")
    typer.echo(f"PyTorch: {torch.__version__}")
    typer.echo(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        typer.echo(f"CUDA Version: {torch.version.cuda}")
        typer.echo(f"GPU: {torch.cuda.get_device_name(0)}")
    typer.echo("")
    typer.echo("Paths:")
    typer.echo(f"  Data Dir: {settings.data_dir}")
    typer.echo(f"  Checkpoint Dir: {settings.checkpoint_dir}")
    typer.echo(f"  Log Dir: {settings.log_dir}")
    typer.echo("")
    typer.echo("Model Settings:")
    typer.echo(f"  Sequence Length: {settings.model_sequence_length}")
    typer.echo(f"  Hidden Size: {settings.model_hidden_size}")
    typer.echo(f"  Num Layers: {settings.model_num_layers}")
    typer.echo("")
    typer.echo("Trading Settings:")
    typer.echo(f"  Symbol: {settings.trading_symbol}")
    typer.echo(f"  Lot Size: {settings.trading_lot_size}")
    typer.echo(f"  Stop Loss: ${settings.trading_stop_loss_usd}")
    typer.echo(f"  Take Profit: ${settings.trading_take_profit_usd}")


if __name__ == "__main__":
    app()

