
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.environment.trading_env import TradingEnvironment
from src.environment.reward_calculator import AdaptiveRewardCalculator, RewardComponents
from src.data.candle import Candle

def test_v2_integration():
    print("STDOUT: Testing V2 Integration...")
    
    # 1. Initialize Adaptive Reward Calculator
    reward_calc = AdaptiveRewardCalculator(normalize_rewards=True)
    print(f"STDOUT: Initialized AdaptiveRewardCalculator: {reward_calc}")
    
    # 2. Inject into Environment
    env = TradingEnvironment(
        input_dim=16,
        reward_calculator=reward_calc
    )
    print(f"STDOUT: Initialized TradingEnvironment with custom reward calculator")
    
    # 3. Verify Injection
    if not isinstance(env.reward_calculator, AdaptiveRewardCalculator):
        print("STDOUT: FAIL: env.reward_calculator is not AdaptiveRewardCalculator")
        return
    else:
        print("STDOUT: PASS: Injection verified")
        
    # 4. Run a step to verify it works
    from datetime import datetime
    
    # Create fake candle
    candle = Candle(
        timestamp=datetime.now(),
        open=20.0, high=21.0, low=19.0, close=20.5, volume=1000.0
    )
    env.receive_candle(candle)
    
    # Fake action
    action = {
        "prediction": np.zeros(16),
        "trading_action": 1 # BUY
    }
    
    print("STDOUT: stepping environment...")
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 5. Check if stats updated (AdaptiveRewardCalculator tracks mean/var)
    stats = env.reward_calculator.get_statistics()
    print(f"STDOUT: Reward Stats after step: {stats}")
    
    if stats['count'] > 0:
        print("STDOUT: PASS: Reward calculator updated stats")
    else:
        print("STDOUT: FAIL: Reward calculator did not update stats")

if __name__ == "__main__":
    try:
        test_v2_integration()
    except Exception as e:
        print(f"STDOUT: Error: {e}")
        import traceback
        traceback.print_exc()
