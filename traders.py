"""
Step 2: simulate traders with different behaviours on the cleaned candle data.

Execution model (identical for every trader; this is what prevents look-ahead bias):
  * A decision is made on a CLOSED candle k (indicators only use data up to and including k).
  * The order is executed at the OPEN of candle k+1.
  * Stop-loss / take-profit / liquidation are checked intrabar with the candle high/low.
    If a stop and a take-profit are both touched in the same candle, the stop is assumed
    to be hit first (conservative).
  * Gaps: if a candle opens beyond a stop level, the fill is at the open, not at the stop.
  * Candle timestamps are assumed to be candle OPEN times.

Position accounting:
  margin   = equity_before_trade * size_frac
  notional = margin * leverage
  pnl      = notional * direction * (exit / entry - 1) - fees,  never worse than -margin
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_clean, load_config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskProfile:
    leverage: float
    size_frac: float                  # fraction of equity used as margin per trade
    stop_loss: float | None = None    # adverse price move that closes the trade, e.g. 0.03
    take_profit: float | None = None  # favourable price move that closes the trade
    max_hold: int | None = None       # maximum holding time in candles


class Trader(ABC):
    name: str = "trader"
    style: str = ""
    profile: RiskProfile

    def prepare(self, df: pd.DataFrame) -> None:
        """Pre-compute indicators. Only causal operations are allowed here
        (rolling / ewm / shift(positive)), so value[i] never depends on data after i."""

    @abstractmethod
    def entry_signal(self, i: int) -> int:
        """Return +1 (long), -1 (short) or 0, using data up to the close of candle i."""

    def exit_signal(self, i: int, direction: int) -> bool:
        """Return True to close the open position, using data up to the close of candle i."""
        return False


# ----------------------------------------------------------------------
# Trader definitions
# ----------------------------------------------------------------------
class ConservativeTrader(Trader):
    """Mean reversion: fades large deviations from a 48h moving average.
    Low leverage, small size, tight stop and take-profit."""
    name = "conservative"
    style = "mean_reversion"
    profile = RiskProfile(leverage=1, size_frac=0.2, stop_loss=0.015, take_profit=0.02, max_hold=48)

    def __init__(self, window: int = 48, entry_z: float = 1.5, exit_z: float = 0.2):
        self.window, self.entry_z, self.exit_z = window, entry_z, exit_z

    def prepare(self, df):
        close = df["close"]
        ma = close.rolling(self.window).mean()
        sd = close.rolling(self.window).std()
        self.z = ((close - ma) / sd).to_numpy()

    def entry_signal(self, i):
        z = self.z[i]
        if np.isnan(z):
            return 0
        if z < -self.entry_z:
            return 1
        if z > self.entry_z:
            return -1
        return 0

    def exit_signal(self, i, direction):
        z = self.z[i]
        return (direction == 1 and z > -self.exit_z) or (direction == -1 and z < self.exit_z)


class TrendFollowingTrader(Trader):
    """EMA(8) / EMA(24) trend follower. Always follows the sign of the EMA spread,
    so it flips direction when the fast EMA crosses the slow one."""
    name = "trend_follower"
    style = "trend_following"
    profile = RiskProfile(leverage=2, size_frac=0.3, stop_loss=0.03)

    def __init__(self, fast: int = 8, slow: int = 24, min_spread: float = 0.0005):
        self.fast, self.slow, self.min_spread = fast, slow, min_spread

    def prepare(self, df):
        close = df["close"]
        fast = close.ewm(span=self.fast, adjust=False).mean()
        slow = close.ewm(span=self.slow, adjust=False).mean()
        self.spread = ((fast - slow) / close).to_numpy()
        self.warmup = self.slow * 3

    def entry_signal(self, i):
        if i < self.warmup or abs(self.spread[i]) < self.min_spread:
            return 0
        return 1 if self.spread[i] > 0 else -1

    def exit_signal(self, i, direction):
        return i >= self.warmup and np.sign(self.spread[i]) == -direction


class HighRiskTrader(Trader):
    """Donchian breakout (24h channel) with high leverage and large size."""
    name = "high_risk"
    style = "breakout_high_leverage"
    profile = RiskProfile(leverage=8, size_frac=0.4, stop_loss=0.05)

    def __init__(self, entry_window: int = 24, exit_window: int = 12):
        self.entry_window, self.exit_window = entry_window, exit_window

    def prepare(self, df):
        self.close = df["close"].to_numpy()
        # shift(1): the channel is built from candles BEFORE the current one
        self.up = df["high"].rolling(self.entry_window).max().shift(1).to_numpy()
        self.down = df["low"].rolling(self.entry_window).min().shift(1).to_numpy()
        self.exit_down = df["low"].rolling(self.exit_window).min().shift(1).to_numpy()
        self.exit_up = df["high"].rolling(self.exit_window).max().shift(1).to_numpy()

    def entry_signal(self, i):
        if np.isnan(self.up[i]):
            return 0
        if self.close[i] > self.up[i]:
            return 1
        if self.close[i] < self.down[i]:
            return -1
        return 0

    def exit_signal(self, i, direction):
        if np.isnan(self.exit_down[i]):
            return False
        if direction == 1:
            return self.close[i] < self.exit_down[i]
        return self.close[i] > self.exit_up[i]


class RandomNoiseTrader(Trader):
    """Enters in a random direction at random times and exits randomly.
    Acts as a 'no skill' baseline: a good scoring system should rank it low."""
    name = "random_noise"
    style = "random"
    profile = RiskProfile(leverage=2, size_frac=0.2, stop_loss=0.03, max_hold=48)

    def __init__(self, seed: int = 42, p_entry: float = 0.05, p_exit: float = 1 / 12):
        self.seed, self.p_entry, self.p_exit = seed, p_entry, p_exit

    def prepare(self, df):
        self.rng = np.random.default_rng(self.seed)

    def entry_signal(self, i):
        if self.rng.random() < self.p_entry:
            return 1 if self.rng.random() < 0.5 else -1
        return 0

    def exit_signal(self, i, direction):
        return self.rng.random() < self.p_exit


class SwingMomentumTrader(Trader):
    """Momentum: trades in the direction of the 12h return when it is strong and
    agrees with the 100h EMA trend. Medium leverage, fixed maximum holding time."""
    name = "swing_momentum"
    style = "momentum_swing"
    profile = RiskProfile(leverage=3, size_frac=0.3, stop_loss=0.025, take_profit=0.06, max_hold=36)

    def __init__(self, roc_window: int = 12, roc_threshold: float = 0.02, trend_span: int = 100):
        self.roc_window, self.roc_threshold, self.trend_span = roc_window, roc_threshold, trend_span

    def prepare(self, df):
        close = df["close"]
        self.close = close.to_numpy()
        self.roc = close.pct_change(self.roc_window).to_numpy()
        self.trend = close.ewm(span=self.trend_span, adjust=False).mean().to_numpy()
        self.warmup = self.trend_span

    def entry_signal(self, i):
        if i < self.warmup or np.isnan(self.roc[i]):
            return 0
        if self.roc[i] > self.roc_threshold and self.close[i] > self.trend[i]:
            return 1
        if self.roc[i] < -self.roc_threshold and self.close[i] < self.trend[i]:
            return -1
        return 0

    def exit_signal(self, i, direction):
        return (direction == 1 and self.roc[i] < 0) or (direction == -1 and self.roc[i] > 0)


def build_traders(seed: int = 42) -> list[Trader]:
    return [
        ConservativeTrader(),
        TrendFollowingTrader(),
        HighRiskTrader(),
        RandomNoiseTrader(seed=seed),
        SwingMomentumTrader(),
    ]


# ----------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------
def simulate_trader(
    trader: Trader,
    df: pd.DataFrame,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    liq_buffer: float = 0.9,
) -> pd.DataFrame:
    """Runs one trader over the candles and returns its trade log."""
    trader.prepare(df)
    ts = df["timestamp"].tolist()
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
    prof = trader.profile

    state = {"equity": float(initial_capital), "pos": None}
    trades: list[dict] = []
    pending_entry = 0
    pending_exit: str | None = None

    def close_position(k: int, price: float, reason: str) -> None:
        pos = state["pos"]
        direction = pos["dir"]
        price_ret = direction * (price / pos["entry_price"] - 1.0)
        gross = pos["notional"] * price_ret
        fee = fee_rate * pos["notional"] * (1.0 + price / pos["entry_price"])
        pnl = max(gross - fee, -pos["margin"])  # a trade can never lose more than its margin
        state["equity"] += pnl
        trades.append(
            {
                "trader": trader.name,
                "style": trader.style,
                "trade_id": f"{trader.name}-{len(trades):04d}",
                "direction": "long" if direction == 1 else "short",
                "signal_time": ts[pos["entry_idx"] - 1],
                "entry_time": ts[pos["entry_idx"]],
                "entry_price": pos["entry_price"],
                "exit_time": ts[k],
                "exit_price": price,
                "exit_reason": reason,
                "hold_bars": k - pos["entry_idx"],
                "leverage": prof.leverage,
                "size_frac": prof.size_frac,
                "margin": pos["margin"],
                "notional": pos["notional"],
                "price_return": price_ret,
                "fee": fee,
                "pnl": pnl,
                "pnl_pct": pnl / pos["equity_before"],
                "equity_after": state["equity"],
            }
        )
        state["pos"] = None

    for k in range(n):
        # 1) Execute orders decided on the previous (closed) candle, at this candle's open
        if pending_exit is not None and state["pos"] is not None:
            close_position(k, o[k], pending_exit)
        pending_exit = None
        if pending_entry != 0 and state["pos"] is None and state["equity"] > 0 and k > 0:
            margin = state["equity"] * prof.size_frac
            state["pos"] = {
                "dir": pending_entry,
                "entry_idx": k,
                "entry_price": o[k],
                "margin": margin,
                "notional": margin * prof.leverage,
                "equity_before": state["equity"],
            }
        pending_entry = 0

        # 2) Intrabar risk exits (stop-loss / liquidation first, then take-profit)
        pos = state["pos"]
        if pos is not None:
            d, ep = pos["dir"], pos["entry_price"]
            liq_move = liq_buffer / prof.leverage
            sl_move = prof.stop_loss if prof.stop_loss is not None else np.inf
            if sl_move <= liq_move:
                adverse_move, adverse_reason = sl_move, "stop_loss"
            else:
                adverse_move, adverse_reason = liq_move, "liquidation"
            adverse_level = ep * (1 - d * adverse_move)
            if d == 1:
                hit, gap = l[k] <= adverse_level, o[k] <= adverse_level
            else:
                hit, gap = h[k] >= adverse_level, o[k] >= adverse_level
            if hit:
                close_position(k, o[k] if gap else adverse_level, adverse_reason)
            elif prof.take_profit is not None:
                tp_level = ep * (1 + d * prof.take_profit)
                if d == 1:
                    tp_hit, tp_gap = h[k] >= tp_level, o[k] >= tp_level
                else:
                    tp_hit, tp_gap = l[k] <= tp_level, o[k] <= tp_level
                if tp_hit:
                    close_position(k, o[k] if tp_gap else tp_level, "take_profit")

        # 3) Decisions at the close of candle k (executed at the open of candle k+1)
        pos = state["pos"]
        if k < n - 1:
            if pos is not None:
                held = k - pos["entry_idx"] + 1
                if prof.max_hold is not None and held >= prof.max_hold:
                    pending_exit = "max_hold"
                elif trader.exit_signal(k, pos["dir"]):
                    pending_exit = "signal"
            if pos is None or pending_exit is not None:
                direction = trader.entry_signal(k)
                if direction != 0:
                    pending_entry = direction
        elif pos is not None:
            close_position(k, c[k], "end_of_data")

    return pd.DataFrame(trades)


def simulate_all(df: pd.DataFrame, sim_cfg: dict) -> pd.DataFrame:
    frames = []
    for trader in build_traders(sim_cfg["seed"]):
        t = simulate_trader(
            trader, df,
            initial_capital=sim_cfg["initial_capital"],
            fee_rate=sim_cfg["fee_rate"],
            liq_buffer=sim_cfg["liquidation_buffer"],
        )
        if len(t) < sim_cfg["min_trades_per_trader"]:
            log.warning("%s produced only %d trades (< %d required)",
                        trader.name, len(t), sim_cfg["min_trades_per_trader"])
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def summarize_trades(trades: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    rows = []
    for name, g in trades.groupby("trader", sort=False):
        rows.append(
            {
                "trader": name,
                "style": g["style"].iloc[0],
                "n_trades": len(g),
                "n_long": int((g["direction"] == "long").sum()),
                "n_short": int((g["direction"] == "short").sum()),
                "win_rate": round(float((g["pnl"] > 0).mean()), 3),
                "total_return": round(float(g["equity_after"].iloc[-1] / initial_capital - 1), 3),
                "avg_hold_bars": round(float(g["hold_bars"].mean()), 1),
                "stops": int((g["exit_reason"] == "stop_loss").sum()),
                "liquidations": int((g["exit_reason"] == "liquidation").sum()),
            }
        )
    return pd.DataFrame(rows)


def build_trader_dataset(cfg_path: str = "config.yaml") -> pd.DataFrame:
    cfg = load_config(cfg_path)
    sim = cfg["simulation"]
    df = load_clean(cfg_path)

    trades = simulate_all(df, sim)
    Path(sim["trades_path"]).parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(sim["trades_path"], index=False)

    summary = summarize_trades(trades, sim["initial_capital"])
    Path(sim["summary_path"]).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(sim["summary_path"], index=False)

    from src.plotting import plot_trader_equity
    plot_trader_equity(trades, sim["initial_capital"], sim["equity_plot_path"])

    print("\n=== Trader summary ===")
    print(summary.to_string(index=False))
    return trades
