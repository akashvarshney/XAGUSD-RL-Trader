import pandas as pd
import argparse
from pathlib import Path
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_dataframe(df, name):
    """Clean and standardize broker CSV exports."""
    # Handle MT5 Tab-separated format if the whole thing is one column
    if len(df.columns) == 1 and '\t' in df.columns[0]:
        # Reload with tab separator
        path = df.index.name # Not reliable, better to handle inside merge_data
        pass 

    # Strip '<' and '>' from column names and lowercase them
    df.columns = [c.replace('<', '').replace('>', '').lower() for c in df.columns]
    
    # Handle MT5 separate date and time columns
    if 'date' in df.columns and 'time' in df.columns:
        logger.info(f"Combining date and time columns for {name}...")
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    else:
        # Common timestamp column names
        ts_cols = ['timestamp', 'datetime', 'date/time']
        ts_col = next((c for c in ts_cols if c in df.columns), None)
        
        if not ts_col:
            logger.error(f"Could not find timestamp column in {name}. Columns: {df.columns.tolist()}")
            return None
        df = df.rename(columns={ts_col: 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Standardize OHLCV names
    ohlcv_map = {
        'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 
        'tickvol': 'volume', 'vol': 'volume', 'volume': 'volume'
    }
    
    for col in df.columns:
        if col in ohlcv_map:
            df = df.rename(columns={col: ohlcv_map[col]})
            
    return df

def merge_data(silver_path, gold_path, output_path):
    """Merge Silver and Gold data with time features."""
    logger.info(f"Loading Silver data: {silver_path}")
    # Try reading with tab first, then comma
    try:
        df_silver = pd.read_csv(silver_path, sep='\t')
        if len(df_silver.columns) < 5:
            df_silver = pd.read_csv(silver_path, sep=',')
    except:
        df_silver = pd.read_csv(silver_path)
        
    df_silver = clean_dataframe(df_silver, "Silver")
    
    logger.info(f"Loading Gold data: {gold_path}")
    try:
        df_gold = pd.read_csv(gold_path, sep='\t')
        if len(df_gold.columns) < 5:
            df_gold = pd.read_csv(gold_path, sep=',')
    except:
        df_gold = pd.read_csv(gold_path)
        
    df_gold = clean_dataframe(df_gold, "Gold")
    
    if df_silver is None or df_gold is None:
        return

    # Keep only necessary columns from Gold
    df_gold = df_gold[['timestamp', 'close']].rename(columns={'close': 'gold_close'})
    
    # Merge on timestamp (Inner join to ensure both exist)
    logger.info("Merging dataframes...")
    merged = pd.merge(df_silver, df_gold, on='timestamp', how='inner')
    
    # Add Time Features
    logger.info("Adding Time Context features...")
    merged['hour_of_day'] = merged['timestamp'].dt.hour
    merged['day_of_week'] = merged['timestamp'].dt.dayofweek
    
    # Sort and reset index
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    
    # Save result
    logger.info(f"Saving merged data to {output_path}")
    merged.to_csv(output_path, index=False)
    
    logger.info("--- MERGE SUMMARY ---")
    logger.info(f"Total Rows: {len(merged)}")
    logger.info(f"Time Range: {merged['timestamp'].min()} to {merged['timestamp'].max()}")
    logger.info(f"New Columns: {['gold_close', 'hour_of_day', 'day_of_week']}")
    print("\nFirst 5 rows of merged data:")
    print(merged.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Silver and Gold broker data for RL training.")
    parser.add_argument("--silver", required=True, help="Path to raw Silver CSV")
    parser.add_argument("--gold", required=True, help="Path to raw Gold CSV")
    parser.add_argument("--output", default="data/historical/XAG_GOLD_PRO.csv", help="Output path")
    
    args = parser.parse_args()
    
    silver_p = Path(args.silver)
    gold_p = Path(args.gold)
    
    if not silver_p.exists() or not gold_p.exists():
        logger.error("One or more input files missing!")
        sys.exit(1)
        
    merge_data(silver_p, gold_p, args.output)
