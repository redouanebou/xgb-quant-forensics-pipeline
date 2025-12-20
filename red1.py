import pandas as pd
import pandas_ta as ta
import numpy as np
import os
from scipy.signal import argrelextrema

RAW_DATA_PATH = r"D:\duka\EURUSD.csv"   #this EURUSD.csv is a high quality data m1 candles path
PROCESSED_DATA_DIR = r"D:\duka\reversal_forensics_data" 
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

TRAIN_END_DATE = '2020-12-31'
VALIDATION_END_DATE = '2023-12-31'
FINAL_TEST_START_DATE = '2024-01-01'

PIP_SIZE = 0.0001
TP_PIPS = 10.0
SL_PIPS = 5.0 
LOOKAHEAD_PERIOD = 60 
REVERSAL_LOOKBACK = 5
EVIDENCE_WINDOW = 25 

def build_base_indicators(df):
    """Just calculates the raw indicator values for the entire dataset first."""
    print("Building the base indicators for forensic analysis...")
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.dropna(inplace=True)
    return df

def run_forensics(df):
    """
    This is the new engine. It finds reversals, labels them, and scrapes the evidence.
    NOW IT CREATES SEPARATE FILES FOR BUYS AND SELLS.
    """
    print("Identifying all potential reversal points...")
    lows_idx = argrelextrema(df['Low'].to_numpy(), np.less_equal, order=REVERSAL_LOOKBACK)[0]
    highs_idx = argrelextrema(df['High'].to_numpy(), np.greater_equal, order=REVERSAL_LOOKBACK)[0]
    
    potential_buys = df.iloc[lows_idx]
    potential_sells = df.iloc[highs_idx]
    
    print(f"Found {len(potential_buys)} potential buy setups and {len(potential_sells)} potential sell setups.")

    buy_evidence = process_reversals(df, potential_buys, 'buy')
    sell_evidence = process_reversals(df, potential_sells, 'sell')
    
    return buy_evidence, sell_evidence

def process_reversals(full_df, candidates_df, direction):
    """Helper function to process either buys or sells."""
    print(f"Processing {direction.upper()} signals...")
    evidence_list = []
    
    tp_target = TP_PIPS * PIP_SIZE
    sl_target = SL_PIPS * PIP_SIZE
    
    for idx, reversal_candle in candidates_df.iterrows():
        future_window_end = idx + pd.Timedelta(minutes=LOOKAHEAD_PERIOD)
        future_df = full_df.loc[idx:future_window_end]
        
        if len(future_df) < 2: continue

        outcome_found = False
        label = 0 
        if direction == 'buy':
            entry_price = reversal_candle['Low']
            tp_level = entry_price + tp_target
            sl_level = entry_price - sl_target
            for _, future_candle in future_df.iloc[1:].iterrows():
                if future_candle['High'] >= tp_level: 
                    label = 1
                    outcome_found = True
                    break
                if future_candle['Low'] <= sl_level: 
                    label = 0
                    outcome_found = True
                    break
        else:
            entry_price = reversal_candle['High']
            tp_level = entry_price - tp_target
            sl_level = entry_price + sl_target
            
            for _, future_candle in future_df.iloc[1:].iterrows():
                if future_candle['Low'] <= tp_level: 
                    label = 1
                    outcome_found = True
                    break
                if future_candle['High'] >= sl_level: 
                    label = 0
                    outcome_found = True
                    break
        
        if outcome_found:
            evidence_start = idx - pd.Timedelta(minutes=EVIDENCE_WINDOW)
            evidence_df = full_df.loc[evidence_start:idx]
            
            if len(evidence_df) < EVIDENCE_WINDOW // 2: continue

            evidence = {
                'timestamp': idx,
                'rsi_mean': evidence_df['RSI_14'].mean(),
                'rsi_std': evidence_df['RSI_14'].std(),
                'macd_hist_sum': evidence_df['MACDh_12_26_9'].sum(),
                'bb_width_mean': evidence_df['BBB_20_2.0'].mean(),
                'atr_mean': evidence_df['ATRr_14'].mean(),
                'label': label
            }
            evidence_list.append(evidence)
            
    if not evidence_list:
        return pd.DataFrame()

    df = pd.DataFrame(evidence_list)
    df = df.set_index(pd.to_datetime(df['timestamp'])).drop('timestamp', axis=1)
    return df

def main():
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    df = pd.read_csv(
        RAW_DATA_PATH,
        names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Tick volume'],
        parse_dates=[['Date', 'Time']],
        index_col='Date_Time',
        skiprows=1
    )
    df.sort_index(inplace=True)
    
    df = build_base_indicators(df)
    
    buy_df, sell_df = run_forensics(df)
    
    for df_to_save, name in [(buy_df, 'buys'), (sell_df, 'sells')]:
        if df_to_save.empty:
            print(f"\nNo valid {name.upper()} data found. Skipping.")
            continue
            
        print(f"\nFinal evidence distribution for {name.upper()}:")
        print(df_to_save['label'].value_counts(normalize=True))
        
        df_to_save.dropna(inplace=True)
        train = df_to_save[:TRAIN_END_DATE]
        validation = df_to_save[TRAIN_END_DATE:VALIDATION_END_DATE]
        final_test = df_to_save[FINAL_TEST_START_DATE:]
        
        print(f"Training set shape for {name.upper()}:   {train.shape}")
        
        train.to_parquet(os.path.join(PROCESSED_DATA_DIR, f'{name}_train.parquet'))
        validation.to_parquet(os.path.join(PROCESSED_DATA_DIR, f'{name}_validation.parquet'))
        final_test.to_parquet(os.path.join(PROCESSED_DATA_DIR, f'{name}_final_test.parquet'))

    print("\nForensics complete.")



if __name__ == "__main__":
    print("Make sure you have 'pandas-ta' and 'scipy'. If not, 'pip install pandas-ta scipy'.")
    main()




