"""Tests for the trader simulation, using synthetic candles (no internet needed)."""
import numpy as np
import pandas as pd
import pytest

from src.traders import build_traders, simulate_trader


def make_candles(n: int = 4384, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = 0.0006 * np.sin(2 * np.pi * np.arange(n) / 700)          # alternating regimes
    ret = drift + rng.standard_t(4, n) * 0.004
    close = 60_000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[60_000], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    ts = pd.date_range("2026-04-01", periods=n, freq="60min", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                         "close": close, "volume": 1.0})


@pytest.fixture(scope="module")
def candles():
    return make_candles()


def test_each_trader_has_enough_trades(candles):
    for trader in build_traders(42):
        trades = simulate_trader(trader, candles)
        assert len(trades) >= 100, f"{trader.name}: {len(trades)} trades"


def test_entries_use_next_open_after_signal(candles):
    open_by_time = candles.set_index("timestamp")["open"]
    for trader in build_traders(42):
        trades = simulate_trader(trader, candles)
        assert (trades["signal_time"] < trades["entry_time"]).all()
        # entry price is exactly the open of the entry candle
        expected = open_by_time.loc[trades["entry_time"]].to_numpy()
        assert np.allclose(trades["entry_price"].to_numpy(), expected)


def test_no_lookahead_truncation_invariance(candles):
    """Trades completed before a cut-off must be identical whether or not
    the future candles exist. If any signal used future data, this would fail."""
    cut = 3000
    short = candles.iloc[:cut].reset_index(drop=True)
    limit = candles["timestamp"].iloc[cut - 50]
    for trader_full, trader_short in zip(build_traders(42), build_traders(42)):
        full = simulate_trader(trader_full, candles)
        part = simulate_trader(trader_short, short)
        a = full[full["exit_time"] <= limit].reset_index(drop=True)
        b = part[part["exit_time"] <= limit].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


def test_loss_never_exceeds_margin(candles):
    for trader in build_traders(42):
        trades = simulate_trader(trader, candles)
        assert (trades["pnl"] >= -trades["margin"] - 1e-9).all()
