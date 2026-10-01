"""
مرحله ۱: دریافت و پاک‌سازی داده کندل از نوبیتکس.

جریان کار:
  fetch_history()  -> داده خام را تکه‌تکه از API می‌گیرد
  clean_candles()  -> تکراری، نامعتبر و کندل‌های جاافتاده را مدیریت می‌کند
  build_dataset()  -> هر دو را اجرا و خروجی را ذخیره می‌کند
"""
import logging
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

log = logging.getLogger(__name__)

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------
# دریافت داده
# ----------------------------------------------------------------------
def fetch_chunk(cfg: dict, start_ts: int, end_ts: int, retries: int = 3) -> pd.DataFrame:
    """یک درخواست به API. زمان‌ها timestamp بر حسب ثانیه (UTC) هستند."""
    url = cfg["base_url"] + cfg["endpoint"]
    params = {
        "symbol": cfg["symbol"],
        "resolution": cfg["resolution"],
        "from": start_ts,
        "to": end_ts,
    }

    data = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            break
        except (requests.RequestException, ValueError) as e:
            log.warning("تلاش %d/%d ناموفق: %s", attempt + 1, retries, e)
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # 1s, 2s, 4s ...

    status = data.get("s")
    if status == "no_data":
        return pd.DataFrame(columns=COLUMNS)
    if status != "ok":
        # فرمت UDF: وقتی خطا باشد معمولاً s="error" و پیام در errmsg است
        raise RuntimeError(f"پاسخ غیرمنتظره از API: {str(data)[:300]}")

    return pd.DataFrame(
        {
            "timestamp": data["t"],
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "close": data["c"],
            "volume": data["v"],
        }
    )


def fetch_history(cfg: dict) -> pd.DataFrame:
    """کل بازه ۶ ماهه را در چند درخواست می‌گیرد و به هم می‌چسباند."""
    end = int(time.time())
    start = end - int(cfg["months"] * 30.44 * 24 * 3600)
    step = int(cfg["chunk_days"] * 24 * 3600)

    parts, cur = [], start
    while cur < end:
        nxt = min(cur + step, end)
        log.info("دریافت %s تا %s",
                 pd.to_datetime(cur, unit="s"), pd.to_datetime(nxt, unit="s"))
        parts.append(fetch_chunk(cfg, cur, nxt))
        cur = nxt
        time.sleep(cfg.get("sleep_between_requests", 0.5))

    parts = [p for p in parts if not p.empty]
    if not parts:
        raise RuntimeError("هیچ داده‌ای دریافت نشد. symbol/resolution/آدرس را چک کن.")
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------
# پاک‌سازی
# ----------------------------------------------------------------------
def clean_candles(raw: pd.DataFrame, resolution_min: int) -> tuple[pd.DataFrame, dict]:
    """
    خروجی: (دیتافریم تمیز، گزارش پاک‌سازی)

    تصمیم‌های طراحی (برای جلسه فنی):
    - تکراری‌ها: چون بازه‌های درخواست‌ها هم‌پوشانی مرزی دارند، آخرین نسخه نگه داشته می‌شود.
    - کندل نامعتبر (قیمت <= 0، high < low، open/close بیرون از [low, high]) حذف می‌شود.
    - کندل جاافتاده: روی یک شبکه زمانی کامل بازسازی می‌شود. قیمت‌ها با close قبلی
      پر می‌شوند (کندل تخت) و volume = 0. ستون is_filled آن‌ها را علامت می‌زند.
      از backward-fill استفاده نمی‌کنیم چون از آینده اطلاعات می‌آورد (look-ahead).
    - Outlier: حذف نمی‌شود (سقوط/جهش شدید در کریپتو واقعی است)؛ فقط گزارش می‌شود.
    """
    report = {"rows_raw": len(raw)}
    df = raw.copy()

    for c in COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n = len(df)
    df = df.dropna()
    report["dropped_nan"] = n - len(df)

    # اگر timestamp میلی‌ثانیه باشد به ثانیه تبدیل کن
    if df["timestamp"].max() > 1e11:
        df["timestamp"] = df["timestamp"] // 1000
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    n = len(df)
    df = df.drop_duplicates(subset="timestamp", keep="last")
    report["dropped_duplicates"] = n - len(df)

    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    bad = (
        (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (df["high"] < df["low"])
        | (df["open"] > df["high"]) | (df["open"] < df["low"])
        | (df["close"] > df["high"]) | (df["close"] < df["low"])
        | (df["volume"] < 0)
    )
    df = df[~bad]
    report["dropped_invalid"] = n - len(df)

    # شبکه زمانی کامل و پر کردن کندل‌های جاافتاده (فقط forward-fill)
    freq = f"{resolution_min}min"
    full_index = pd.date_range(df["timestamp"].iloc[0], df["timestamp"].iloc[-1], freq=freq)
    df = df.set_index("timestamp").reindex(full_index)
    df.index.name = "timestamp"
    df["is_filled"] = df["close"].isna()
    report["missing_candles_filled"] = int(df["is_filled"].sum())

    df["close"] = df["close"].ffill()
    for c in ["open", "high", "low"]:
        df[c] = df[c].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)

    # گزارش outlier (حذف نمی‌کنیم)
    ret = df["close"].pct_change()
    z = (ret - ret.mean()) / ret.std()
    report["extreme_return_candles(|z|>6)"] = int((z.abs() > 6).sum())

    report["rows_clean"] = len(df)
    report["start"] = str(df.index[0])
    report["end"] = str(df.index[-1])
    return df.reset_index(), report


# ----------------------------------------------------------------------
def build_dataset(cfg_path: str = "config.yaml") -> pd.DataFrame:
    cfg = load_config(cfg_path)["data"]

    raw = fetch_history(cfg)
    Path(cfg["raw_path"]).parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(cfg["raw_path"], index=False)

    clean, report = clean_candles(raw, int(cfg["resolution"]))
    Path(cfg["clean_path"]).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(cfg["clean_path"], index=False)

    print("\n=== گزارش پاک‌سازی ===")
    for k, v in report.items():
        print(f"{k}: {v}")
    return clean


def load_clean(cfg_path: str = "config.yaml") -> pd.DataFrame:
    """برای مراحل بعدی: خواندن داده تمیز از دیسک."""
    cfg = load_config(cfg_path)["data"]
    return pd.read_csv(cfg["clean_path"], parse_dates=["timestamp"])
