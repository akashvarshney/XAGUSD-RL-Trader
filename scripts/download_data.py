import yfinance as yf
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_data(symbol="SI=F", start_date="2024-03-15", interval="1h"):
    """
    Download historical data from Yahoo Finance.
    
    Args:
        symbol: Ticker symbol (SI=F for Silver Futures)
        start_date: Start date string
        interval: Data interval (e.g., "1h", "1d", "1m")
    """
    logger.info(f"Downloading data for {symbol}...")
    
    # Download data
    df = yf.download(symbol, start=start_date, interval=interval, progress=False)
    
    if df.empty:
        logger.error("No data downloaded!")
        return
        
    # Reset index to make Date/Datetime a column
    df = df.reset_index()
    
    # Flatten multi-level columns if present (yfinance update)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Rename columns to match project expectation (lowercased)
    df.columns = [c.lower() for c in df.columns]
    
    # Ensure standard OHLCV columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Rename 'datetime' or 'date' to 'timestamp'
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'timestamp'})
    elif 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
        
    # Verify columns
    available_cols = [c for c in required_cols if c in df.columns]
    if len(available_cols) < 5:
        logger.error(f"Missing columns. Found: {df.columns}")
        return

    # Select and order columns
    df = df[['timestamp'] + required_cols]
    
    # Save to CSV
    output_dir = Path("data/historical")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "XAGUSD_1H.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Data saved to {output_file}")
    logger.info(f"Total rows: {len(df)}")
    print(df.head())

if __name__ == "__main__":
    download_data()
