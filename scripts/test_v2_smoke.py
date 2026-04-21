
import sys
from pathlib import Path
import time

print("STDOUT: Script started")
sys.stdout.flush()

try:
    import numpy as np
    print("STDOUT: Numpy imported")
    sys.stdout.flush()
    
    import torch
    print(f"STDOUT: Torch imported. Version: {torch.__version__}")
    sys.stdout.flush()
    
    from src.environment.trading_env import TradingEnvironment
    print("STDOUT: TradingEnvironment imported")
    sys.stdout.flush()
    
    env = TradingEnvironment(input_dim=16)
    print("STDOUT: Env initialized")
    sys.stdout.flush()
    
except Exception as e:
    print(f"STDOUT: Error: {e}")
    sys.stdout.flush()
