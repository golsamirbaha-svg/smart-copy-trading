from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; we only save files
import matplotlib.pyplot as plt


def plot_trader_equity(trades, initial_capital: float, path: str) -> None:
    """Equity curve of every simulated trader (log scale), built from their trade logs."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, g in trades.groupby("trader", sort=False):
        x = [g["entry_time"].iloc[0]] + g["exit_time"].tolist()
        y = [initial_capital] + g["equity_after"].tolist()
        ax.step(x, y, where="post", label=name)
    ax.set_yscale("log")
    ax.set_title("Simulated traders: equity curves")
    ax.set_ylabel("Equity (USDT, log scale)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
