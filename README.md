# 🤖 XGBoost Quant Forensics Pipeline

This project is a 2-stage quantitative research pipeline. It's designed to find high-probability trade setups by performing "forensic analysis" on past reversals and then using an AI "bouncer" to filter out bad signals.

---

## 1. Stage 1: The "Forensics" Script (`darwin.py`)

This script acts like a detective. Instead of trying to predict every candle, it focuses only on high-potential reversal points (local highs/lows).

* **Finds Reversals:** Uses `scipy.signal.argrelextrema` to find all local highs and lows in the entire dataset.
* **Labels Outcomes:** It time-travels to each reversal and checks what happened next (within a 60-minute window). It labels the reversal as:
    * `1 (Good Trade)`: If it hit a 1:2 Risk/Reward Take Profit.
    * `0 (Bad Trade)`: If it hit the Stop Loss.
* **Scrapes Evidence:** For every *labeled* reversal, it goes back in time to the **25-minute window *before*** the signal. It scrapes data (RSI, MACD, BBands) from this "evidence window" to see what the market looked like just before a good or bad trade.
* **Saves Data:** It saves this "evidence" (the features) and the "outcome" (the label) into clean `.parquet` files, creating a perfect dataset for training an AI.

## 2. Stage 2: The "Bouncer" Backtester (`label2.py`)

This script simulates a live trading bot that uses the AI model trained on the data from Stage 1.

* **Assumes All Signals:** The backtester *assumes* every reversal signal is a potential trade.
* **Consults AI:** For each signal, it gathers the "evidence" (the features from the 25-minute window) and feeds it to a pre-trained `XGBoost` model.
* **Acts as "Bouncer":**
    * If the AI predicts `1` (Good Trade), the bot "bounces" the signal and **approves** the trade.
    * If the AI predicts `0` (Bad Trade), the bot **rejects** the trade.
* **Honest Simulation:** It runs a candle-by-candle simulation (no look-ahead bias) to see how this AI-filtered strategy would have *actually* performed.
