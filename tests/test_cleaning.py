"""تست پاک‌سازی با داده ساختگی (بدون نیاز به اینترنت)."""
import pandas as pd
from src.data_loader import clean_candles


def test_clean_handles_dups_gaps_invalid():
    t0 = 1_700_000_000 - (1_700_000_000 % 3600)
    rows = []
    for i in range(10):
        if i == 4:           # کندل جاافتاده
            continue
        p = 100 + i
        rows.append([t0 + i * 3600, p, p + 1, p - 1, p, 5.0])
    rows.append(rows[0])                                   # تکراری
    rows.append([t0 + 20 * 3600, 100, 90, 110, 100, 1.0])  # high < low
    raw = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

    df, rep = clean_candles(raw, 60)
    assert rep["dropped_duplicates"] == 1
    assert rep["dropped_invalid"] == 1
    assert rep["missing_candles_filled"] == 1
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    filled = df[df["is_filled"]].iloc[0]
    assert filled["volume"] == 0 and filled["close"] == 103   # ffill از کندل قبلی، نه آینده
