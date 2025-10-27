import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import random
import MetaTrader5 as mt5
from datetime import datetime
import pytz
import xgboost as xgb
import os
import pandas_ta as ta
import time

SYMBOL = "EURUSD"
BACKTEST_START_DATE = datetime(2024, 1, 1, tzinfo=pytz.UTC)
BACKTEST_END_DATE = datetime(2025, 9, 23, tzinfo=pytz.UTC)
TIMEFRAME = mt5.TIMEFRAME_M1
MODEL_DIR = r"."
INITIAL_BALANCE = 10000.0
RISK_PER_TRADE_PERCENT = 0.01  #mean 0.01% of account balance in SL
COMMISSION_PER_LOT = 7.0
PIP_SIZE = 0.0001
TP_PIPS = 10.0
SL_PIPS = 5.0
REALISTIC_MAX_LOT_SIZE = 100.0 
REVERSAL_LOOKBACK = 5
EVIDENCE_WINDOW = 25

class LiveTickerBacktester:
    def __init__(self):
        self.balance = INITIAL_BALANCE
        self.equity = INITIAL_BALANCE
        self.trade_history = []
        self.is_trade_open = False
        self.current_trade = {}
        
        self.buy_bouncer = self.load_bouncer('buys')
        self.sell_bouncer = self.load_bouncer('sells')
        
        self.data = self.load_and_prepare_data()
        if self.data is None: return
            
        self.signals = self.precompute_signals()

    def load_bouncer(self, specialist_type):
        model_path = os.path.join(MODEL_DIR, f'{specialist_type}_bouncer.json')
        print(f"Mongo Tom: Loading the {specialist_type.upper()} bouncer...")
        try:
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            print(f"{specialist_type.upper()} bouncer is online.")
            return model
        except Exception as e:
            print(f"Could not load the {specialist_type.upper()} bouncer. Error: {e}")
            return None

    def load_and_prepare_data(self):
        if not mt5.initialize(): 
            print("MT5 initialize() failed.")
            return None
            
        print(f"Fetching broker data for the live ticker...")
        rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, BACKTEST_START_DATE, BACKTEST_END_DATE)
        mt5.shutdown()
        if rates is None or len(rates) == 0: return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'time': 'Date_Time', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Tick volume'}, inplace=True)
        df.set_index('Date_Time', inplace=True)
        
        print("Building base indicators")
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.atr(length=14, append=True)
        return df.dropna()

    def precompute_signals(self):
        print("Finding all potential money signals...")
        lows_idx = argrelextrema(self.data['Low'].to_numpy(), np.less_equal, order=REVERSAL_LOOKBACK)[0]
        highs_idx = argrelextrema(self.data['High'].to_numpy(), np.greater_equal, order=REVERSAL_LOOKBACK)[0]
        buy_signals = self.data.iloc[lows_idx].index
        sell_signals = self.data.iloc[highs_idx].index
        signals = pd.Series(dtype=object)
        signals = pd.concat([signals, pd.Series('BUY', index=buy_signals)])
        signals = pd.concat([signals, pd.Series('SELL', index=sell_signals)])
        signals = signals[~signals.index.duplicated(keep='first')].sort_index()
        return signals

    def get_forensic_evidence(self, timestamp):
        evidence_start = timestamp - pd.Timedelta(minutes=EVIDENCE_WINDOW)
        if evidence_start < self.data.index[0]: return None
        evidence_df = self.data.loc[evidence_start:timestamp]
        
        if len(evidence_df) < EVIDENCE_WINDOW // 2: return None

        evidence_data = {
            'rsi_mean': evidence_df['RSI_14'].mean(), 'rsi_std': evidence_df['RSI_14'].std(),
            'macd_hist_sum': evidence_df['MACDh_12_26_9'].sum(), 'bb_width_mean': evidence_df['BBB_20_2.0'].mean(),
            'atr_mean': evidence_df['ATRr_14'].mean()
        }
        feature_names = self.buy_bouncer.get_booster().feature_names
        return pd.DataFrame([evidence_data], columns=feature_names)

    def run_simulation(self):
        if self.data is None or self.buy_bouncer is None or self.sell_bouncer is None: return
            
        print("\n" + "="*50)
        print(" Live Simulation Ticker ENGAGED")
        print("="*50 + "\n")
        time.sleep(2)

        for timestamp, candle in self.data.iterrows():
            current_time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            if self.is_trade_open:
                self.check_trade_exit(candle, timestamp)

            if not self.is_trade_open:
                if timestamp in self.signals.index:
                    signal_type = self.signals.loc[timestamp]
                    print(f"[{current_time_str}] Potential {signal_type} signal detected. Consulting AI bouncer...")
                    
                    evidence = self.get_forensic_evidence(timestamp)
                    if evidence is None:
                        print(f"[{current_time_str}] Insufficient historical data to form evidence. Signal ignored.")
                        continue

                    bouncer = self.buy_bouncer if signal_type == 'BUY' else self.sell_bouncer
                    prediction = bouncer.predict(evidence)[0]

                    if prediction == 1: 
                        print(f"[{current_time_str}] Bouncer says GO! Preparing to open {signal_type} trade.")
                        try:
                            next_candle_timestamp = self.data.index[self.data.index.get_loc(timestamp) + 1]
                            next_candle_open = self.data.loc[next_candle_timestamp, 'Open']
                            self.open_trade(signal_type, next_candle_open, next_candle_timestamp)
                        except IndexError: 
                            print(f"[{current_time_str}] Signal at the end of data. Cannot open trade.")
                            continue
                    else: 
                        print(f"[{current_time_str}] Bouncer says Trade rejected.")
                else:
                    print(f"\r[{current_time_str}] Scanning... No signal.", end="")

        if self.is_trade_open:
            self.close_trade(self.data.iloc[-1]['Close'], self.data.index[-1], "END_OF_BACKTEST")

        self.print_final_report()

    def open_trade(self, direction, entry_price_raw, timestamp):
        spread = random.uniform(0.0, 0.3) * PIP_SIZE
        entry_price = entry_price_raw + spread if direction == 'BUY' else entry_price_raw - spread
        
        risk_amount = self.equity * RISK_PER_TRADE_PERCENT
        value_per_pip_per_lot = 10 
        loss_per_lot = (SL_PIPS * value_per_pip_per_lot) + COMMISSION_PER_LOT
        if loss_per_lot <= 0: return

        lot_size = round(risk_amount / loss_per_lot, 2)
        lot_size = min(lot_size, REALISTIC_MAX_LOT_SIZE)
        if lot_size < 0.01: return
        
        sl = entry_price - (SL_PIPS * PIP_SIZE) if direction == 'BUY' else entry_price + (SL_PIPS * PIP_SIZE)
        tp = entry_price + (TP_PIPS * PIP_SIZE) if direction == 'BUY' else entry_price - (TP_PIPS * PIP_SIZE)
        
        self.is_trade_open = True
        self.current_trade = {'open_time': timestamp, 'direction': direction, 'entry_price': entry_price, 'sl': sl, 'tp': tp, 'lot_size': lot_size}
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] >>> TRADE OPENED: {direction} {lot_size} lots @ {entry_price:.5f} | SL: {sl:.5f}, TP: {tp:.5f}")


    def check_trade_exit(self, candle, timestamp):
        trade = self.current_trade
        exit_price, reason = (None, "")
        if trade['direction'] == 'BUY':
            if candle['Low'] <= trade['sl']: exit_price, reason = trade['sl'], "SL_HIT"
            elif candle['High'] >= trade['tp']: exit_price, reason = trade['tp'], "TP_HIT"
        else:
            if candle['High'] >= trade['sl']: exit_price, reason = trade['sl'], "SL_HIT"
            elif candle['Low'] <= trade['tp']: exit_price, reason = trade['tp'], "TP_HIT"
        if exit_price: self.close_trade(exit_price, timestamp, reason)

    def close_trade(self, close_price, timestamp, reason):
        trade = self.current_trade
        pips_moved = (close_price - trade['entry_price']) / PIP_SIZE
        if trade['direction'] == 'SELL': pips_moved = -pips_moved
        gross_pnl = pips_moved * 10 * trade['lot_size']
        commission_cost = COMMISSION_PER_LOT * trade['lot_size']
        net_pnl = gross_pnl - commission_cost
        
        self.balance += net_pnl; self.equity = self.balance
        
        log_entry = {'Profit': net_pnl}
        self.trade_history.append(log_entry)
        
        self.is_trade_open = False
        self.current_trade = {}
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] <<< TRADE CLOSED: {reason} | Profit: ${net_pnl:,.2f} | New Balance: ${self.balance:,.2f}")


    def print_final_report(self):
        if not self.trade_history: return
        print("\n" + "="*50); print("FINAL SIMULATION Report"); print("="*50)
        trades_df = pd.DataFrame(self.trade_history)
        wins = trades_df[trades_df['Profit'] > 0]
        total_trades = len(trades_df)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        gross_profit = wins['Profit'].sum()
        gross_loss = abs(trades_df[trades_df['Profit'] <= 0]['Profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        print(f"{'Final Balance':<20}: ${self.balance:,.2f}"); print(f"{'Total Trades':<20}: {total_trades}"); print(f"{'Win Rate':<20}: {win_rate:.2%}"); print(f"{'Profit Factor':<20}: {profit_factor:.2f}"); print("=" * 50)

if __name__ == "__main__":
    backtester = LiveTickerBacktester()
    backtester.run_simulation()

