import logging
import os
from typing import Dict

import pandas as pd

from config import *
from utils import TechnicalIndicators

logger = logging.getLogger(__name__)


class RangingStrategy:
    """Mean-reversion strategy for ranging markets using Bollinger Bands and RSI."""

    def __init__(self, mt5_connector):
        self.mt5 = mt5_connector
        self.name = "Ranging"
        # Parameters (can be moved to config if needed)
        self.bb_period = int(os.getenv("RANGING_BB_PERIOD", "20"))
        self.bb_std = float(os.getenv("RANGING_BB_STD", "2.0"))
        self.rsi_period = int(os.getenv("RANGING_RSI_PERIOD", "14"))
        self.rsi_low = int(os.getenv("RANGING_RSI_LOW", "35"))
        self.rsi_high = int(os.getenv("RANGING_RSI_HIGH", "65"))
        self.atr_period = int(os.getenv("RANGING_ATR_PERIOD", "14"))
        self.min_atr_ratio = float(os.getenv("RANGING_MIN_ATR_RATIO", "0.0015"))

    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze for mean-reversion entries inside a range.

        Returns a dict with keys: signal ('buy'|'sell'|'no_signal'), confidence, entry_price, stop_loss, take_profit.
        """
        try:
            if df is None or len(df) < max(self.bb_period, self.rsi_period) + 5:
                return {"signal": "no_signal", "reason": "Insufficient data"}

            close = df["close"]
            high = df["high"]
            low = df["low"]

            # Indicators
            bb_high_series, bb_low_series, bb_mid_series = TechnicalIndicators.bollinger_bands(
                close, self.bb_period, self.bb_std
            )
            if bb_high_series is None or bb_low_series is None:
                return {"signal": "no_signal", "reason": "BB not available"}
            bb_high = float(bb_high_series.iloc[-1])
            bb_low = float(bb_low_series.iloc[-1])

            rsi = TechnicalIndicators.rsi(close, self.rsi_period)
            if rsi is None or len(rsi) < 1:
                return {"signal": "no_signal", "reason": "RSI not available"}
            rsi_now = float(rsi.iloc[-1])

            atr = TechnicalIndicators.atr(high, low, close, self.atr_period)
            atr_now = float(atr.iloc[-1]) if atr is not None else None

            price = float(close.iloc[-1])

            # Simple range-ness heuristic: price within bands and ATR small relative to price
            if atr_now is None or atr_now / max(price, 1e-6) > self.min_atr_ratio:
                return {"signal": "no_signal", "reason": "ATR too high (trending)"}

            # Entry logic: fade extremes with RSI confirmation
            signal = "no_signal"
            confidence = 0.0
            entry = stop = tp = None

            # Potential buy near lower band with RSI not oversold-extreme
            if price <= bb_low and rsi_now >= self.rsi_low:
                signal = "buy"
                # Use ATR for SL/TP sizing
                sl_distance = 1.2 * atr_now
                tp_distance = 1.8 * atr_now
                entry = price
                stop = max(price - sl_distance, 0)
                tp = price + tp_distance
                # Confidence scales with proximity to band and RSI middle
                band_prox = min(1.0, max(0.0, (bb_low - price) / max(atr_now, 1e-6) + 0.5))
                rsi_conf = 1.0 - abs(50 - rsi_now) / 50.0
                confidence = max(0.0, min(100.0, 60 * band_prox + 40 * rsi_conf))

            # Potential sell near upper band with RSI not overbought-extreme
            elif price >= bb_high and rsi_now <= self.rsi_high:
                signal = "sell"
                sl_distance = 1.2 * atr_now
                tp_distance = 1.8 * atr_now
                entry = price
                stop = price + sl_distance
                tp = price - tp_distance
                band_prox = min(1.0, max(0.0, (price - bb_high) / max(atr_now, 1e-6) + 0.5))
                rsi_conf = 1.0 - abs(50 - rsi_now) / 50.0
                confidence = max(0.0, min(100.0, 60 * band_prox + 40 * rsi_conf))

            if signal == "no_signal":
                return {"signal": "no_signal", "reason": "No mean-reversion edge"}

            return {
                "signal": signal,
                "confidence": float(confidence),
                "entry_price": float(entry) if entry is not None else None,
                "stop_loss": float(stop) if stop is not None else None,
                "take_profit": float(tp) if tp is not None else None,
                "reason": "Bollinger mean-reversion with RSI",
            }
        except Exception as e:
            logger.exception("Error in RangingStrategy.analyze")
            return {"signal": "error", "reason": str(e)}


