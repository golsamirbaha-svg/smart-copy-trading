# Smart Copy Trading - Technical Challenge

A runnable prototype of a **Smart Copy Trading** system. Unlike plain copy trading, it does not copy
every trade of a trader blindly: for each trade it decides **whether to copy it, how large the position
should be, and how confident it is**, using the trader's recent performance, the trade's risk, current
market volatility and the copier's capital. It is compared against **Naive copy trading** (copy everything)
in a leak-free backtest on 6 months of BTC/USDT data.

> Traders are **simulated** (rule-based strategies run on real historical candles), not real accounts.

---

## 1. How to run

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python main.py --step data         # 1. download + clean 6 months of 1h BTC/USDT candles (Nobitex)
python main.py --step traders      # 2. simulate 5 traders (>=100 trades each)
python main.py --step evaluate     # 3. metrics + rolling scores
python main.py --step engine       # 4. (optional) copy decisions per trade
python main.py --step risk         # 5. (optional) how often the risk limits bind
python main.py --step backtest     # 6. Naive vs Smart: tuning on train, final result on test
python main.py --step diagnostics  #    why the backtest looks the way it does
python main.py --step robustness   # 7. fee / slippage / delay sensitivity
python -m pytest tests -q          # automated tests (no internet needed)
```

**Reproducibility:** the `data` step downloads the *last 6 months relative to the day you run it*, so a
re-download shifts the window and changes the numbers. The exact dataset used for the reported results is
committed in `data/processed/btcusdt_1h_clean.csv`; to reproduce the results, **skip the `data` step**.
All parameters live in `config.yaml`.

## 2. Repository structure

```
config.yaml            all parameters (fees, limits, grids, paths)
main.py                entry point (--step ...)
src/
  data_loader.py       download from Nobitex UDF endpoint + cleaning
  traders.py           5 simulated traders + execution simulator
  evaluation.py        metrics, composite score, rolling (past-only) scores
  copy_engine.py       Smart Copy Engine: copy / skip, confidence, position size
  risk.py              risk limits and drawdown circuit breaker
  backtest.py          Naive vs Smart backtest, train/test split, tuning
  robustness.py        fee / slippage / delay sweeps
  diagnostics.py       analysis of the results (not used for decisions)
  plotting.py          figures
tests/                 automated tests (cleaning, no look-ahead, risk limits, accounting, ...)
data/processed/        clean candles, trader trades, scores, backtest logs
results/               tables and figures
```

## 3. Design

### 3.1 Market data
- Source: Nobitex public API, TradingView-UDF endpoint `GET https://apiv2.nobitex.ir/market/udf/history`
  (symbol `BTCUSDT`, resolution 60 minutes). Requests are made in 15-day chunks because the API returns
  at most about 500 candles per request (with 30-day chunks 26% of the candles were silently missing,
  which is how this limit was discovered).
- Cleaning: duplicates removed (chunk boundaries overlap), invalid candles removed (price <= 0,
  high < low, open/close outside [low, high]), missing candles rebuilt on a full time grid with
  **forward-fill only** (flat candle, volume 0, flagged by `is_filled`). Backward-fill is never used
  because it would import future information. Extreme-return candles are reported but not removed
  (real crashes/spikes).

### 3.2 Simulated traders (`src/traders.py`)
| Trader | Behaviour | Leverage | Size (of equity) | Stop / TP |
|---|---|---|---|---|
| `conservative` | mean reversion: fades z-score > 1.5 vs 48h average | 1x | 20% | 1.5% / 2% |
| `trend_follower` | EMA 8 / EMA 24, follows the sign of the spread | 2x | 30% | 3% / - |
| `high_risk` | 24h Donchian breakout | 8x | 40% | 5% / - |
| `random_noise` | random entries and exits (no-skill baseline) | 2x | 20% | 3% / - |
| `swing_momentum` | trades 12h return > 1.2% in the direction of the EMA-100 trend | 3x | 30% | 2.5% / 6% |

Execution model (same for every trader): decision on a **closed** candle, fill at the **next open**;
stop/take-profit/liquidation checked intrabar with high/low (stop wins if both are touched in one candle);
gap-through fills at the open; a trade can never lose more than its margin; fee 0.1% per side.
The program warns if a trader has fewer than 100 trades (`results/trader_summary.csv`).

### 3.3 Trader evaluation (`src/evaluation.py`)
Metrics on realized (closed-trade) equity: **Return, Win Rate, Maximum Drawdown, Volatility, Sharpe**
(volatility and Sharpe from daily returns, annualized with 365, risk-free rate 0).

Composite score in [0, 1] (0.5 = neutral), every component mapped with an **absolute** rule (not a
rank among traders):

| Component | Weight | Mapping |
|---|---|---|
| Sharpe | 0.35 | logistic(Sharpe) |
| Max drawdown | 0.25 | 1 - min(drawdown / 30%, 1) |
| Return | 0.15 | logistic(return / 5%) |
| Volatility | 0.15 | 1 - min(volatility / 100%, 1) |
| Win rate | 0.10 | win rate |

With fewer than 30 closed trades in the window the score is shrunk toward 0.5 (low confidence).
Two outputs are kept strictly apart: a **full-period table** (descriptive only, never used for decisions)
and **rolling 60-day scores** computed daily from trades **closed at or before that time**.

### 3.4 Smart Copy Engine (`src/copy_engine.py`)
The copier acts 1 candle (1 hour) after the trader. For each trade the first failing rule skips it:
`trade_already_closed` -> `market_warmup` -> `no_history` (< 10 recent trades) -> `low_score`
(< `min_trader_score`) -> `excessive_leverage` (> 5x) -> `extreme_volatility` (> 3x normal) ->
`low_confidence`. Otherwise:

- **Confidence** = trader score x `1 / sqrt(trade leverage)`
- **Size** = `target_daily_risk / daily_volatility x confidence x equity` (volatility targeting: positions
  shrink automatically when the market becomes more volatile)

### 3.5 Risk management (`src/risk.py`)
`approved size = min(desired, every cap)`, caps measured on **current** equity:
Max Position Size 25%, Max Allocation per trader 40%, Max portfolio Leverage 2x, minimum order 10 USDT.
**Maximum Drawdown circuit breaker:** at -20% from the equity peak, new positions are blocked and open
copies are flattened; copying resumes after 7 days and the peak is reset. The limit that decided each
size ("binding limit") is logged.

### 3.6 Backtest (`src/backtest.py`)
Both strategies copy the same traders under identical conditions (delay, fee 0.1%, slippage 0.05%).
The copier's PnL is computed from candle prices at the **copier's own** fill times; the trader only
supplies timing and direction.
- **Naive:** copies every trade; capital split equally between the 5 traders; each trade mirrors the
  trader's own exposure (`size_frac x leverage`); no filters, no limits.
- **Smart:** engine + risk manager.
- Equity is marked to market every candle; open positions are closed at the end of each period.

## 4. Data leakage / look-ahead bias

1. **Next-open execution:** a decision made on candle *k* is filled at the open of *k+1*; indicators only
   use closed candles (causal `rolling`/`ewm`, channels are `shift(1)`).
2. **Past-only scores:** the score at time *t* only uses trades closed at or before *t*, over the last 60 days.
3. **Observable information only:** the copier cannot see the future exit of a trade; an exit that is
   intrabar (stop, take-profit) becomes visible only when its candle has closed.
4. **Time-based split, never random:** 30-day warm-up (nothing is copied) -> **train** (first 60% of the
   rest: Smart's parameters are tuned here only) -> **test** (last 40%: run once with the frozen parameters).
5. **Robustness keeps parameters frozen** (not re-tuned per scenario).
6. **Tests** that would fail on any leak: replacing all candles after a date with random garbage must not
   change the equity curve before that date (naive, smart, and delay 0); truncating the future must not
   change past scores, trades or decisions; a shorter backtest must be a prefix of a longer one.

## 5. Results (real data: BTC/USDT 1h, Nobitex)

Periods: warm-up 2026-04-03 -> 05-03, **train** 05-03 -> 08-03, **test** 08-03 -> 10-03.
Smart parameters tuned on train only: `min_trader_score = 0.5`, `target_daily_risk = 0.005`.
Initial capital 10,000 USDT, fee 0.1% per side, slippage 0.05%, delay 1 hour.

### Test period (final result)
| Strategy | Return | Sharpe | Max drawdown | Volatility | Win rate | Trades | Avg leverage | Fees (USDT) | Final equity |
|---|---|---|---|---|---|---|---|---|---|
| Naive copy | +1.5% | 0.44 | 11.1% | 31.7% | 32.5% | 243 | 0.68 | 863 | 10,155 |
| **Smart copy** | -6.9% | -4.11 | 8.1% | 10.2% | 24.6% | 126 | 0.21 | 362 | 9,308 |
| Smart (untuned, default params) | -13.1% | -6.62 | 13.5% | 12.3% | 31.6% | 190 | 0.44 | 787 | 8,692 |
| Buy & hold BTC (reference) | +34.7% | 4.43 | 7.7% | 41.5% | - | 1 | - | 0 | 13,468 |

### Train period (used for tuning, so optimistic for Smart)
| Strategy | Return | Sharpe | Max drawdown | Volatility | Win rate | Trades | Avg leverage | Fees (USDT) | Final equity |
|---|---|---|---|---|---|---|---|---|---|
| Naive copy | -14.6% | -2.13 | 20.9% | 27.3% | 37.6% | 367 | 0.76 | 1,319 | 8,544 |
| Smart copy | -4.7% | -3.86 | 5.3% | 4.9% | 39.3% | 117 | 0.14 | 333 | 9,531 |
| Buy & hold BTC (reference) | -20.3% | -2.48 | 29.5% | 33.6% | - | 1 | - | 0 | 7,972 |

Equity curves: `results/equity_curves.png` (full tables: `results/backtest_comparison.md`).

![Equity curves](results/equity_curves.png)

### Honest reading of the result
- On the test period **Smart did not beat Naive on return or Sharpe**. It did what a risk filter should
  do on risk: lower drawdown (8.1% vs 11.1%), about one third of the volatility, fewer trades and fees.
  In the (falling) train period it also lost less than Naive.
- Diagnostics (`--step diagnostics`):
  - The rolling score was **not a stable predictor**: Spearman correlation between a trader's score and
    its next-7-day return was +0.22 on train and -0.13 on test (only 5 traders and overlapping windows,
    so neither number is statistically strong).
  - Naive's test profit came almost entirely from `high_risk` (+223 USDT of +155 total). Smart skips that
    trader because its 8x leverage exceeds the 5x limit, which was costly in a strongly rising market.
  - Smart's largest loss came from `conservative` (-388 USDT, 23 trades, 22% win rate).
- Design weakness: the score rewards low exposure (small positions mean low volatility and low drawdown)
  even without skill; the no-skill `random_noise` trader had the highest score (0.80) at the end of the data.
- No parameter or rule was changed after seeing test results.

### Robustness (fee / slippage / delay, test period)
Tables: `results/robustness.md` and `results/robustness.csv`; figure: `results/robustness.png`.

![Robustness](results/robustness.png)

_Key takeaways: to be added after running `python main.py --step robustness`._

## 6. Assumptions and limitations
- Traders are simulated rule-based strategies, not real traders; they have no guaranteed persistent skill.
- Candle timestamps are assumed to be candle **open** times (TradingView UDF convention).
- Fee 0.1% per side and slippage 0.05% per fill are assumptions (real exchange fees can be higher; see the
  robustness sweep). Slippage is a fixed fraction, not dependent on order size.
- Hourly candles: intrabar order of events is unknown; when stop and take-profit are both touched in one
  candle the stop is assumed to hit first.
- Risk limits use each position's notional **at entry**, not its current value (simplification).
- The test period is only about two months of a mostly rising market and covers a single asset;
  the result is one sample, not proof that either strategy is better in general.
- Only 5 traders: score-quality statistics are weak.

## 7. Possible improvements (not done, to avoid tuning on test data)
- Normalize the score by the trader's exposure/risk so low-exposure traders are not rewarded for it.
- Allow higher leverage only for traders with a long, stable track record.
- Use many more traders and several market regimes (walk-forward evaluation).
