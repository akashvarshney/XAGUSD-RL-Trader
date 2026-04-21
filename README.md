# XAGUSD Deep RL Trading System

A Deep Reinforcement Learning trading system for XAGUSD (Silver) using LSTM with Attention mechanism and PPO algorithm.

## Features

- **LSTM-Attention Model**: Processes 120 historical candles with temporal attention for better pattern recognition
- **Hybrid PPO Agent**: Predicts next candle (continuous) and trading action (discrete) simultaneously
- **Real-time Trading**: Integrates with Match-Trader API for live trading
- **Web Dashboard**: React-based monitoring interface with real-time updates
- **Dual Training Modes**: Pre-training on historical CSV data + online learning during live trading

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LSTM-Attention Encoder                   │
│    Input: [batch, 120, 5] OHLCV → Output: [batch, 256]     │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │ Prediction  │  │   Action    │  │   Value     │
     │    Head     │  │    Head     │  │    Head     │
     │  (5 vals)   │  │ (4 logits)  │  │ (1 scalar)  │
     └─────────────┘  └─────────────┘  └─────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- uv (Python package manager)
- Node.js 18+ (for dashboard)

### Installation

```bash
# Clone the repository
cd xagusd-rl-trader

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Copy environment template
cp env.example .env
# Edit .env with your settings
```

### Pre-training on Historical Data

```bash

# Place your CSV file in data/historical/XAGUSD_1M.csv
# CSV format: timestamp,open,high,low,close,volume

# Run pre-training
python scripts/pretrain.py --csv data/historical/XAGUSD_1M.csv --epochs 100
python scripts/pretrain.py --csv data/historical/XAG_GOLD_PRO.csv --timesteps 1000000

```

### Running the Server

```bash
# Start the backend server
python -m src.main serve

# Or with uvicorn directly
uvicorn src.server.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

### Dashboard

```bash
cd dashboard
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

## Project Structure

```
xagusd-rl-trader/
├── src/
│   ├── config/          # Configuration and settings
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Neural network architectures
│   ├── environment/     # Gymnasium trading environment
│   ├── agent/           # PPO agent and training
│   ├── server/          # FastAPI backend
│   └── utils/           # Logging, checkpoints, metrics
├── dashboard/           # React frontend
├── scripts/             # Training and utility scripts
├── data/historical/     # CSV data files
├── checkpoints/         # Model checkpoints
├── logs/
│   ├── tensorboard/     # TensorBoard logs
│   └── trades/          # Trade history CSVs
└── tests/               # Unit tests
```

## Configuration

All settings can be configured via environment variables or `.env` file:

| Variable                    | Description                   | Default |
| --------------------------- | ----------------------------- | ------- |
| `TRADING_LOT_SIZE`        | Position size in lots         | 0.3     |
| `TRADING_STOP_LOSS_USD`   | Stop loss in USD              | 300     |
| `TRADING_TAKE_PROFIT_USD` | Take profit in USD            | 500     |
| `TRADING_MAX_LOSS_USD`    | Max loss before episode fails | 1000    |
| `MODEL_SEQUENCE_LENGTH`   | Number of candles to process  | 120     |
| `MODEL_HIDDEN_SIZE`       | LSTM hidden dimension         | 256     |
| `TRAINING_LEARNING_RATE`  | PPO learning rate             | 0.0003  |

See `env.example` for all available options.

## API Endpoints

### Agent Control

- `POST /api/agent/start` - Start live trading
- `POST /api/agent/stop` - Stop trading
- `GET /api/agent/status` - Get agent status

### Training

- `POST /api/training/pretrain` - Start pre-training
- `POST /api/training/stop` - Stop training
- `GET /api/training/progress` - Get training metrics

### Data

- `GET /api/data/candles` - Get historical candles
- `GET /api/data/trades` - Get trade history
- `GET /api/data/metrics` - Get performance metrics

### WebSocket

- `WS /ws/live` - Real-time updates stream

## Reward Function

```python
# Prediction penalty (MAPE)
prediction_penalty = mean_absolute_percentage_error(predicted, actual)

# Penalty scales with total loss
penalty_weight = 1 + (total_loss / 500)

# PnL reward
pnl_reward = realized_pnl + unrealized_pnl_delta

# Total reward
reward = -prediction_penalty * penalty_weight + pnl_reward * 0.01
```

## Episode Termination

- **Failure**: `total_loss + unrealized_pnl > 1000 USD`
- **Normal end**: End of CSV data (pre-training) or manual stop (live)

## License

MIT License
