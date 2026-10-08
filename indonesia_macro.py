import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv
import datetime
import io
import time
import re
import requests
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

# ── Setup ─────────────────────────────────────────────────────────────────────
# Indonesia counterpart of usa_macro.py / eu_macro.py - same layout, theme, helpers and
# performance design (one cached full-history download per request, sliced to the date range
# in memory, plus a background prefetch). Sources, all free and verified live (2026-10-08):
#   IMF Data API (api.imf.org, SDMX 3.0, no key) - balance of payments (full current/financial
#     account detail, quarterly), international reserves (monthly), merchandise trade totals and
#     by partner (monthly), CPI, BI policy/deposit/lending rates, IDR/USD, broad money, real GDP
#     by expenditure (quarterly).
#   BIS (stats.bis.org, no key) - BI policy rate (daily), effective exchange rates, property prices.
#   UN Comtrade public API (no key, ~1 request at a time) - exports/imports by HS chapter.
#   yfinance - Jakarta Composite, LQ45, IDR, EIDO, large caps, commodities.
# Not available free without a key: the core / administered / volatile-food inflation split,
# CPI by expenditure group after the 2025 rebasing, and unemployment (all BPS - optional
# BPS_API_KEY, see Prices tab). Bank Indonesia's downloadable SEKI .xls tables were checked
# and are a stale archive (e.g. BoP summary ends 2010), so they aren't used.
load_dotenv()
BPS_API_KEY = os.getenv("BPS_API_KEY")

IMF_BASE = "https://api.imf.org/external/sdmx/3.0"
BIS_BASE = "https://stats.bis.org/api/v2/data/dataflow/BIS"
COMTRADE_URL = "https://comtradeapi.un.org/public/v1/preview/C/M/HS"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

# ── Background prefetch (same design as usa_macro.py) ───────────────────────
# Only side-effect-free (_*_raw) loaders run in these threads - they have no ScriptRunContext,
# so an st.warning inside would be silently dropped instead of shown in its tab.
class _NoCtxWarningFilter(logging.Filter):
    def filter(self, record):
        return not (threading.current_thread().name.startswith("idn_bg")
                    and "missing ScriptRunContext" in record.getMessage())

@st.cache_resource
def _bg_pool():
    for name in ("streamlit.runtime.scriptrunner_utils.script_run_context",
                 "streamlit.runtime.scriptrunner.script_run_context"):
        logging.getLogger(name).addFilter(_NoCtxWarningFilter())
    return ThreadPoolExecutor(max_workers=10, thread_name_prefix="idn_bg")

@st.cache_resource
def _bg_submitted():
    return {}

def _bg_call(fn, *args):
    try:
        fn(*args)
    except Exception:
        pass

def prefetch(fn, *args, every=1800):
    key = (fn.__name__, args)
    seen = _bg_submitted()
    now = time.time()
    if now - seen.get(key, 0) < every:
        return
    seen[key] = now
    _bg_pool().submit(_bg_call, fn, *args)

# ── HTTP with per-host concurrency caps, retries and a short failure memo ────
@st.cache_resource
def _http_slots():
    return {"imf": threading.BoundedSemaphore(4), "bis": threading.BoundedSemaphore(4),
            "other": threading.BoundedSemaphore(6)}

@st.cache_resource
def _recent_failures():
    return {}

FAILURE_MEMO_SECS = 120

def _guarded(key, fn):
    """Fail fast (for 2 minutes) on a request that just failed - e.g. in the background
    prefetch - instead of making the tab sit through another full round of retries."""
    rec = _recent_failures().get(key)
    if rec and time.time() - rec[0] < FAILURE_MEMO_SECS:
        raise rec[1]
    try:
        out = fn()
    except Exception as e:
        _recent_failures()[key] = (time.time(), e)
        raise
    _recent_failures().pop(key, None)
    return out

def _http_get(url, params=None, slot="other", tries=3, headers=None, timeout=(10, 45)):
    last = None
    for attempt in range(tries):
        try:
            with _http_slots()[slot]:
                r = requests.get(url, params=params, headers=headers or UA, timeout=timeout)
            if r.status_code == 200 and r.content.strip():
                return r
            if r.status_code in (400, 404):
                raise ValueError(f"HTTP {r.status_code}: {r.text[:120]}")
            last = RuntimeError(f"HTTP {r.status_code}")
        except ValueError:
            raise
        except requests.RequestException as e:
            last = e
        if attempt < tries - 1:
            time.sleep(1.5 * (attempt + 1))
    raise last

def _to_timestamps(periods) -> pd.DatetimeIndex:
    """Period strings -> period-START timestamps: 2026-10-05, 2026-08, 2026-M08 (IMF),
    2026-Q2, 2026."""
    s = pd.Series(periods).astype(str)
    first = s.iloc[0]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", first):
        return pd.DatetimeIndex(pd.to_datetime(s))
    if re.fullmatch(r"\d{4}-M\d{2}", first):
        return pd.DatetimeIndex(pd.to_datetime(s.str.replace("-M", "-", regex=False) + "-01"))
    if re.fullmatch(r"\d{4}-\d{2}", first):
        return pd.DatetimeIndex(pd.to_datetime(s + "-01"))
    if re.fullmatch(r"\d{4}-Q\d", first):
        return pd.PeriodIndex(s, freq="Q").to_timestamp()
    if re.fullmatch(r"\d{4}", first):
        return pd.DatetimeIndex(pd.to_datetime(s + "-01-01"))
    return pd.DatetimeIndex(pd.to_datetime(s))

def _slice(s, start, end):
    """Inclusive date slice that keeps the monthly/quarterly observation whose PERIOD contains
    `start` (same rule as FRED/ECB startPeriod - see usa_macro.py)."""
    if s is None or len(s) == 0:
        return s
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    idx = s.index
    pos = idx.searchsorted(pd.Timestamp(start)) if start else 0
    if start and pos > 0 and len(idx) > 2:
        step_days = float(np.median(np.diff(pd.DatetimeIndex(idx[-25:]).values).astype("timedelta64[D]").astype(float)))
        if step_days >= 25:
            months = max(1, round(step_days / 30.44))
            if idx[pos - 1] + pd.DateOffset(months=months) > pd.Timestamp(start):
                pos -= 1
    out = s.iloc[pos:]
    return out.loc[:pd.Timestamp(end)] if end else out

# ── IMF Data API ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=21600, show_spinner=False)
def _imf_raw(flow: str, key: str, since: str = "2005") -> dict:
    """One IMF SDMX 3.0 request -> {"DIM=CODE|DIM=CODE|...": Series}. `+` inside a key
    dimension selects several codes in one call (verified). Pure - raises on failure."""
    def run():
        r = _http_get(f"{IMF_BASE}/data/dataflow/IMF.STA/{flow}/+/{key}", params={"c[TIME_PERIOD]": f"ge:{since}"},
                      headers={**UA, "Accept": "application/json"}, slot="imf")
        j = r.json()
        struct = j["data"]["structures"][0]
        sdims = struct["dimensions"]["series"]
        odim = struct["dimensions"]["observation"][0]["values"]
        out = {}
        for skey, sval in j["data"]["dataSets"][0].get("series", {}).items():
            codes = {sdims[i]["id"]: sdims[i]["values"][int(k)]["id"] for i, k in enumerate(skey.split(":"))}
            obs = {odim[int(k)]["value"]: v[0] for k, v in sval.get("observations", {}).items()
                   if v and v[0] is not None}
            if not obs:
                continue
            s = pd.Series({p: float(v) for p, v in obs.items()})
            s.index = _to_timestamps(s.index.values)
            out["|".join(f"{d}={c}" for d, c in codes.items())] = s.sort_index()
        return out
    return _guarded(("imf", flow, key, since), run)

def imf_pick(raw: dict, **codes) -> pd.Series:
    """The one series in an _imf_raw() result whose dimensions match **codes."""
    want = [f"{d}={c}" for d, c in codes.items()]
    for k, s in raw.items():
        parts = k.split("|")
        if all(w in parts for w in want):
            return s
    return pd.Series(dtype=float)

def imf_load(flow, key, since="2005", what=""):
    try:
        return _imf_raw(flow, key, since)
    except Exception as e:
        st.warning(f"Could not load {what or flow} from the IMF: {e}")
        return {}

# ── BIS ───────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=21600, show_spinner=False)
def _bis_raw(flow: str, key: str) -> pd.Series:
    def run():
        r = _http_get(f"{BIS_BASE}/{flow}/1.0/{key}", params={"format": "csv"}, slot="bis")
        df = pd.read_csv(io.StringIO(r.text), usecols=["TIME_PERIOD", "OBS_VALUE"])
        s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").values, index=_to_timestamps(df["TIME_PERIOD"].values))
        return s.sort_index().dropna()
    return _guarded(("bis", flow, key), run)

def bis(flow, key, label, start=None, end=None, warn=True) -> pd.Series:
    try:
        return _slice(_bis_raw(flow, key), start, end).rename(label)
    except Exception as e:
        if warn:
            st.warning(f"Could not load {label} from the BIS: {e}")
        return pd.Series(dtype=float, name=label)

# ── UN Comtrade (exports/imports by HS chapter) ───────────────────────────────
# The public preview endpoint needs no key but allows ~1 request at a time (parallel calls get
# HTTP 429 instantly - verified) and takes ~5-10s per month. So the months are filled by ONE
# background worker into a process-wide store, and the Trade tab renders whatever has arrived
# instead of blocking the page for ~4 minutes on a cold start.
COMTRADE_MONTHS = 14  # covers the latest reported month AND the same month a year earlier (YoY), given the ~3-4 month lag
HS_CHAPTERS = {
    "27": "Mineral fuels (coal, oil, gas)", "15": "Animal & vegetable fats (palm oil)", "72": "Iron & steel (incl. ferro-nickel)",
    "75": "Nickel & articles", "26": "Ores, slag & ash", "71": "Precious metals & stones", "85": "Electrical machinery",
    "84": "Machinery & appliances", "87": "Vehicles & parts", "38": "Misc. chemical products", "40": "Rubber",
    "44": "Wood", "48": "Paper", "64": "Footwear", "61": "Knitted apparel", "62": "Woven apparel", "39": "Plastics",
    "29": "Organic chemicals", "10": "Cereals", "23": "Food residues & animal feed", "03": "Fish & seafood",
    "16": "Meat & fish preparations", "74": "Copper", "76": "Aluminium", "17": "Sugars", "12": "Oil seeds",
    "52": "Cotton", "73": "Articles of iron & steel", "90": "Optical & medical instruments", "30": "Pharmaceuticals",
    "28": "Inorganic chemicals", "31": "Fertilisers", "89": "Ships & boats", "88": "Aircraft", "25": "Salt, sulphur, cement",
}

@st.cache_resource
def _comtrade_store():
    return {"data": {}, "lock": threading.Lock(), "running": False, "errors": {}}

def _comtrade_periods():
    # Indonesia reports to Comtrade with a ~3-month lag; start the window 3 months back.
    last = (pd.Timestamp.today().normalize() - pd.DateOffset(months=3)).to_period("M")
    return [str(p).replace("-", "") for p in pd.period_range(last - (COMTRADE_MONTHS - 1), last, freq="M")][::-1]

def _comtrade_fetch_month(period, flow):
    for attempt in range(4):
        r = requests.get(COMTRADE_URL, params={"reporterCode": 360, "period": period, "flowCode": flow,
                                               "cmdCode": "AG2", "partnerCode": 0}, headers=UA, timeout=(10, 90))
        if r.status_code == 429:
            time.sleep(3 + attempt * 2)
            continue
        r.raise_for_status()
        rows = r.json().get("data", []) or []
        return pd.Series({x["cmdCode"]: x["primaryValue"] for x in rows if x.get("partnerCode") == 0}, dtype=float)
    raise RuntimeError("Comtrade rate limit")

def _comtrade_worker():
    """Single sequential worker: newest months first, exports and imports, skipping months
    already stored (refreshed after 24h). Empty responses (month not yet reported) are retried
    on the next run."""
    store = _comtrade_store()
    try:
        for period in _comtrade_periods():
            for flow in ("X", "M"):
                key = (period, flow)
                have = store["data"].get(key)
                if have is not None and time.time() - have[0] < 86400:
                    continue
                try:
                    s = _comtrade_fetch_month(period, flow)
                    if not s.empty:
                        with store["lock"]:
                            store["data"][key] = (time.time(), s)
                except Exception as e:
                    store["errors"][key] = str(e)
                time.sleep(1.2)
    finally:
        store["running"] = False

def start_comtrade_worker():
    store = _comtrade_store()
    with store["lock"]:
        if store["running"]:
            return
        store["running"] = True
    threading.Thread(target=_comtrade_worker, name="idn_bg_comtrade", daemon=True).start()

def comtrade_frames():
    """(exports, imports): DataFrames indexed by month, one column per HS chapter (USD), from
    whatever the worker has fetched so far; plus (#months loaded, #months wanted)."""
    store = _comtrade_store()
    with store["lock"]:
        items = dict(store["data"])
    out = {}
    for flow in ("X", "M"):
        rows = {pd.Timestamp(f"{p[:4]}-{p[4:]}-01"): s for (p, f), (_, s) in items.items() if f == flow}
        out[flow] = pd.DataFrame(rows).T.sort_index() if rows else pd.DataFrame()
    return out["X"], out["M"], len(items), COMTRADE_MONTHS * 2

# ── Indonesian government bond yields (PHEI IGSYC, recorded daily) ────────────
# No free source has HISTORICAL SBN yields (verified 2026-10-08: AsianBondsOnline shows only
# Bloomberg snapshots "due to license agreements", OECD's Indonesian long-term rate stopped at
# Dec 2024, IMF has none). PHEI - the official Indonesia Bond Pricing Agency - publishes the
# fitted government yield curve (IGSYC) every day on a public page, today + yesterday only.
# So the dashboard records it: each run reads the page (cached 1h) and upserts those two days
# into data/idr_sbn_yields.csv, building a history from 2026-10-08 onward. Note: Streamlit
# Cloud's disk is wiped on redeploy, so the long history accumulates on local runs.
PHEI_URL = "https://www.phei.co.id/Data/HPW-dan-Imbal-Hasil"
SBN_TENORS = [1, 2, 3, 5, 7, 10, 15, 20, 30]
YIELD_STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "idr_sbn_yields.csv")
ID_MONTHS = {"januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6, "juli": 7, "agustus": 8,
             "september": 9, "oktober": 10, "november": 11, "desember": 12, "january": 1, "february": 2, "march": 3,
             "may": 5, "june": 6, "july": 7, "august": 8, "october": 10, "december": 12}

@st.cache_data(ttl=3600, show_spinner=False)
def _phei_curve_raw() -> pd.DataFrame:
    """PHEI's IGSYC curve -> 2-row DataFrame (index: curve date and the previous business day,
    columns: tenor in years). Pure - raises on failure."""
    def run():
        r = _http_get(PHEI_URL, slot="other")
        m = re.search(r"Indonesia Government Securities Yield Curve.{0,400}?per\s*(\d{1,2})-([A-Za-z]+)-(\d{4})", r.text, re.S)
        if not m:
            raise ValueError("PHEI page layout changed (curve date not found)")
        d = pd.Timestamp(int(m.group(3)), ID_MONTHS[m.group(2).lower()], int(m.group(1)))
        tables = [t for t in pd.read_html(io.StringIO(r.text), decimal=",", thousands=".")
                  if list(t.columns)[:3] == ["Tenor Year", "Today", "Yesterday"]]
        if not tables:
            raise ValueError("PHEI page layout changed (curve table not found)")
        curve = pd.concat(tables).set_index("Tenor Year").astype(float)
        out = pd.DataFrame({d: curve["Today"], d - pd.offsets.BDay(1): curve["Yesterday"]}).T
        out = out[[t for t in out.columns if float(t) in SBN_TENORS]]
        out.columns = [f"{int(float(t))}Y" for t in out.columns]
        return out.sort_index()
    return _guarded(("phei",), run)

@st.cache_resource
def _yield_store_lock():
    return threading.Lock()

def load_sbn_history(record=True):
    """Recorded SBN curve history (date x tenor). With record=True, first upserts PHEI's latest
    two days into the CSV (warning shown if PHEI or the disk write fails)."""
    hist = pd.DataFrame()
    if os.path.exists(YIELD_STORE):
        try:
            hist = pd.read_csv(YIELD_STORE, index_col=0, parse_dates=True)
        except Exception as e:
            st.warning(f"Could not read the recorded SBN yield history ({YIELD_STORE}): {e}")
    if record:
        try:
            latest = _phei_curve_raw()
            with _yield_store_lock():
                merged = pd.concat([hist[~hist.index.isin(latest.index)], latest]).sort_index() if not hist.empty else latest
                if hist.empty or not merged.equals(hist):
                    os.makedirs(os.path.dirname(YIELD_STORE), exist_ok=True)
                    merged.round(4).to_csv(YIELD_STORE, index_label="date")
            hist = merged
        except Exception as e:
            st.warning(f"Could not record today's SBN curve from PHEI: {e}")
    return hist.sort_index()

# ── Bank Indonesia indicator pages (fresher than the IMF mirror) ──────────────
# The IMF republishes BI/BPS data with a 1-2 month lag. Two of BI's public indicator pages are
# plain HTML tables that run ahead of it (verified 2026-10-09): monthly CPI YoY (10 latest
# months) and the monetary-indicators table, whose memo item is official FX reserves (IRFCL,
# USD mn) for the latest two month-ends. They only EXTEND the IMF series - months the IMF
# already has keep the IMF value (they matched on the overlap: July CPI 2.88% in both).
BI_INFLATION_URL = "https://www.bi.go.id/id/statistik/indikator/data-inflasi.aspx"
BI_MONETARY_URL = "https://www.bi.go.id/id/statistik/indikator/indikator-moneter.aspx"

@st.cache_data(ttl=21600, show_spinner=False)
def _bi_inflation_raw() -> pd.Series:
    def run():
        r = _http_get(BI_INFLATION_URL, slot="other")
        t = pd.read_html(io.StringIO(r.text))[0]
        out = {}
        for period, val in zip(t.iloc[:, 0].astype(str), t.iloc[:, 1].astype(str)):
            m = re.match(r"\s*([A-Za-z]+)\s+(\d{4})", period)
            v = re.search(r"-?\d+(?:[.,]\d+)?", val)
            if m and v and m.group(1).lower() in ID_MONTHS:
                out[pd.Timestamp(int(m.group(2)), ID_MONTHS[m.group(1).lower()], 1)] = float(v.group(0).replace(",", "."))
        if not out:
            raise ValueError("BI inflation page layout changed")
        return pd.Series(out).sort_index()
    return _guarded(("bi_inflation",), run)

@st.cache_data(ttl=21600, show_spinner=False)
def _bi_reserves_raw() -> pd.Series:
    """Official FX reserves (IRFCL, USD bn) for the month-ends on BI's monetary-indicators page."""
    def run():
        r = _http_get(BI_MONETARY_URL, slot="other")
        t = pd.read_html(io.StringIO(r.text))[0]
        label_cols = [c for c in t.columns if str(c).startswith("Items")]
        row = t[t[label_cols].astype(str).apply(lambda col: col.str.contains("Cadangan Devisa", case=False)).any(axis=1)]
        if row.empty:
            raise ValueError("BI monetary-indicators page layout changed (reserves row not found)")
        out = {}
        for c in t.columns:
            try:
                d = pd.to_datetime(str(c), format="%d %b %Y")
            except (ValueError, TypeError):
                continue
            out[pd.Timestamp(d.year, d.month, 1)] = float(row.iloc[0][c]) / 1e3
        return pd.Series(out).sort_index()
    return _guarded(("bi_reserves",), run)

def _extend_with(base: pd.Series, newer: pd.Series) -> pd.Series:
    """base + only the newer source's months after base's last month."""
    if newer is None or newer.empty:
        return base
    if base.empty:
        return newer
    return pd.concat([base, newer[newer.index > base.index.max()]]).sort_index()

def cpi_yoy_series(cpi_raw) -> pd.Series:
    imf = imf_pick(cpi_raw, TYPE_OF_TRANSFORMATION="YOY_PCH_PA_PT")
    try:
        bi_ = _bi_inflation_raw()
    except Exception:
        bi_ = pd.Series(dtype=float)
    return _extend_with(imf, bi_)

def reserves_total_series(il_raw) -> pd.Series:
    # Gold at NATIONAL valuation (TRGNV) - BI's own published basis: Aug-2026 IMF 146.49bn vs BI 146.51bn,
    # whereas the market-value series (TRGMV, 146.82bn) would put a fake ~0.3bn step at the splice.
    imf = imf_pick(il_raw, INDICATOR="TRGNV_REVS") / 1e9
    try:
        bi_ = _bi_reserves_raw()
    except Exception:
        bi_ = pd.Series(dtype=float)
    return _extend_with(imf, bi_)

# ── FRED (US Treasury yields for the IDR-vs-US rate beta) ─────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY")
UST_SERIES = {"1Y": "DGS1", "5Y": "DGS5", "10Y": "DGS10"}

@st.cache_data(ttl=3600, show_spinner=False)
def _fred_raw(series_id: str) -> pd.Series:
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY not set")
    def run():
        # No browser User-Agent here: FRED's API silently stalls requests that send one (verified
        # 2026-10-08: browser UA -> read timeout, default requests UA -> 0.3s).
        r = _http_get("https://api.stlouisfed.org/fred/series/observations", slot="other",
                      headers={"Accept": "application/json"},
                      params={"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json",
                              "observation_start": "2005-01-01"})
        obs = r.json()["observations"]
        s = pd.Series({o["date"]: o["value"] for o in obs})
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    return _guarded(("fred", series_id), run)

# ── yfinance (same lock + raw/wrapper split as usa_macro.py) ─────────────────
@st.cache_resource
def _yf_lock():
    return threading.Lock()

def _yf_download(*args, **kwargs):
    with _yf_lock():  # yf.download is not thread-safe (module-global yfinance.shared._DFS)
        return yf.download(*args, **kwargs)

def _yf_retry(fn, retries=3, backoff=1.5):
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (attempt + 1))
    raise last_exc if last_exc else RuntimeError("yfinance call failed")

@st.cache_data(ttl=3600, show_spinner=False)
def _yf_batch_raw(tickers: tuple, start: str) -> pd.DataFrame:
    return _yf_retry(lambda: _yf_download(list(tickers), start=start, progress=False,
                                          auto_adjust=True, group_by="ticker"))

def fetch_yf_batch(tickers_map: dict, start: str) -> dict:
    tickers = list(tickers_map)
    try:
        hist = _yf_batch_raw(tuple(tickers), start)
    except Exception as e:
        st.warning(f"Could not load {', '.join(tickers_map.values())} from Yahoo Finance: {e}")
        return {t: pd.DataFrame() for t in tickers}
    out = {}
    for t, label in tickers_map.items():
        try:
            close = hist["Close"] if len(tickers) == 1 else hist[t]["Close"]
            s = close.dropna()
            df = pd.DataFrame({label: s.values}, index=pd.to_datetime(s.index).tz_localize(None))
            df.index.name = "date"
            out[t] = df
        except (KeyError, TypeError):
            out[t] = pd.DataFrame()
    return out

st.set_page_config(page_title="Indonesia Macro Dashboard", layout="wide", page_icon="🇮🇩")

st.markdown("""
<style>
  .stApp { background-color: #0e1117; }
  .block-container { padding-top: 1rem; }
  .metric-card {
    background: #161b26; border: 1px solid #2a2f3e;
    border-radius: 8px; padding: 12px 16px; text-align: center;
  }
  .metric-label { font-size: 0.68rem; color: #8a94a6; letter-spacing: 0.08em;
                  text-transform: uppercase; margin-bottom: 3px; }
  .metric-value { font-size: 1.2rem; font-weight: 700; white-space: nowrap; }
  .metric-delta { font-size: 0.72rem; margin-top: 2px; }
  .metric-z { font-size: 0.62rem; margin-top: 4px; padding-top: 4px;
              border-top: 1px solid #2a2f3e; display: flex; justify-content: center; gap: 8px; }
  .positive { color: #26a69a; }
  .negative { color: #ef5350; }
  .neutral  { color: #e0e0e0; }
  .section-header {
    font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: #8a94a6; margin: 1.2rem 0 0.5rem;
    border-bottom: 1px solid #2a2f3e; padding-bottom: 4px;
  }
</style>
""", unsafe_allow_html=True)

TEMPLATE   = "plotly_dark"
PAPER_BG   = "#0e1117"
PLOT_BG    = "#161b26"
GRID_COLOR = "#2a2f3e"
CRISIS_COLOR = "rgba(180,60,60,0.12)"
PALETTE = ["#42a5f5", "#ef5350", "#26a69a", "#ff9800", "#ab47bc", "#eda100", "#8bc34a", "#4dd0e1", "#f06292",
           "#90a4d4", "#c85fd6", "#6bbf8f"]

# Indonesia has no official business-cycle dating, so shading marks the major stress episodes
# instead: the 2013 taper tantrum, the 2018 EM sell-off, the 2020 COVID recession and the 2026
# Strait of Hormuz closure (Indonesia is a net oil importer).
STRESS_EPISODES = [("2013-05-22", "2013-09-30", "Taper tantrum"), ("2018-04-01", "2018-10-31", "EM sell-off"),
                   ("2020-03-01", "2020-09-30", "COVID"), ("2026-03-02", None, "Hormuz closure")]

def episodes_in(start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    for a, b, _ in STRESS_EPISODES:
        a, b = pd.Timestamp(a), pd.Timestamp(b) if b else e
        if b >= s and a <= e:
            out.append((max(a, s), min(b, e)))
    return out

def add_episodes(fig, episodes, rows=None, cols=None):
    for a, b in episodes:
        if rows and cols:
            for row, col in zip(rows, cols):
                fig.add_vrect(x0=a, x1=b, fillcolor=CRISIS_COLOR, layer="below", line_width=0, row=row, col=col)
        else:
            fig.add_vrect(x0=a, x1=b, fillcolor=CRISIS_COLOR, layer="below", line_width=0)
    return fig

def add_ma_overlays(fig, full_df, col_name, start, end, row=None, col=None,
                    color_200d="#8a94a6", color_200w="#4fc3f7"):
    if full_df.empty or col_name not in full_df.columns:
        return fig
    s = full_df[col_name].dropna()
    ma_200d = s.rolling(200).mean()
    ma_200w = s.resample("W").last().dropna().rolling(200).mean()
    ma_200d = ma_200d[(ma_200d.index >= start) & (ma_200d.index <= end)]
    ma_200w = ma_200w[(ma_200w.index >= start) & (ma_200w.index <= end)]
    kwargs = dict(row=row, col=col) if row is not None and col is not None else {}
    if not ma_200d.empty:
        fig.add_trace(go.Scatter(x=ma_200d.index, y=ma_200d.values, name="200D MA",
                                 line=dict(color=color_200d, width=1.1, dash="dot")), **kwargs)
    if not ma_200w.empty:
        fig.add_trace(go.Scatter(x=ma_200w.index, y=ma_200w.values, name="200W MA",
                                 line=dict(color=color_200w, width=1.3, dash="dash")), **kwargs)
    return fig

def base_layout(title="", height=480):
    return dict(
        template=TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)),
        height=height, margin=dict(l=50, r=50, t=45, b=30),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(size=10)),
        xaxis=dict(gridcolor=GRID_COLOR), yaxis=dict(gridcolor=GRID_COLOR),
    )

def dual_axis_layout(title, y1_title, y2_title, height=480):
    layout = base_layout(title, height)
    layout.update(yaxis=dict(title=y1_title, gridcolor=GRID_COLOR),
                  yaxis2=dict(title=y2_title, overlaying="y", side="right", gridcolor=GRID_COLOR))
    return layout

# ── 1-year z-score hover on every time-series line ──────────────────────────
# Chart data is clipped with ~13 months of warm-up history before the selected start date
# (_clip / _clip_mkt), so finalize_fig() can compute each point's trailing-1-year z-score on
# real history - not just the visible window - then trims every trace back to [START, END].
# The window is calendar-based ("365D"), so it is right for daily, weekly, monthly and
# quarterly series alike. Moving-average overlays are skipped (a z-score of an MA is noise).
ZSCORE_WARMUP_MONTHS = 13
_MA_NAME = re.compile(r"\bMA\b")

def _is_datetime_axis(x) -> bool:
    arr = np.asarray(x)
    if arr.dtype.kind == "M":
        return True
    return arr.dtype == object and len(arr) > 0 and isinstance(arr[0], (pd.Timestamp, datetime.datetime, datetime.date, np.datetime64))

def _z1y_time(y: pd.Series) -> np.ndarray:
    out = np.full(len(y), np.nan)
    valid = ~np.isnan(y.values)
    s = y[valid]
    if len(s) < 4 or not s.index.is_monotonic_increasing or s.index.has_duplicates:
        return out
    # observations per year from the series' own density (trading-day series have ~255, not 365)
    span_days = max((s.index[-1] - s.index[0]).days, 1)
    per_year = max(1.0, len(s) / span_days * 365.0)
    roll = s.rolling("365D", min_periods=max(3, int(per_year * 0.75)))
    z = (s - roll.mean()) / roll.std()
    out[valid] = z.replace([np.inf, -np.inf], np.nan).values   # flat stretches (e.g. an unchanged policy rate) have no z
    return out

def finalize_fig(fig):
    s0, e0 = pd.Timestamp(START), pd.Timestamp(END) + pd.Timedelta(days=1)
    for tr in fig.data:
        x = getattr(tr, "x", None)
        if x is None or len(x) == 0 or not _is_datetime_axis(x):
            continue
        xi = pd.DatetimeIndex(pd.to_datetime(np.asarray(x)))
        name = tr.name or ""
        is_line = tr.type == "scatter" and (tr.mode is None or "lines" in tr.mode)
        if is_line and tr.customdata is None and tr.y is not None and not _MA_NAME.search(name):
            y = pd.Series(pd.to_numeric(pd.Series(np.asarray(tr.y)), errors="coerce").values, index=xi, dtype=float)
            tr.customdata = _z1y_time(y)
            tr.hovertemplate = f"%{{x|%Y-%m-%d}}<br>{name}: %{{y:,.2f}}<br>Z (1Y): %{{customdata:.2f}}<extra></extra>"
        keep = np.where((xi >= s0) & (xi < e0))[0]
        if len(keep) == len(xi):
            continue
        n = len(xi)
        def take(v):
            if v is None or isinstance(v, str):
                return v
            arr = list(v)
            return [arr[i] for i in keep] if len(arr) == n else v
        tr.x = list(xi[keep])          # keep real timestamps (a list(np.datetime64) would serialize as ns integers)
        tr.y = take(tr.y)
        tr.customdata = take(tr.customdata)
        tr.text = take(tr.text)
        if getattr(tr, "marker", None) is not None and tr.marker.color is not None and not isinstance(tr.marker.color, str):
            tr.marker.color = take(tr.marker.color)
    return fig

def show_chart(fig, key):
    st.plotly_chart(finalize_fig(fig), use_container_width=True, key=key)

def csv_download(df, label: str):
    if df is None:
        return
    if isinstance(df.index, pd.DatetimeIndex) and len(df):
        df = df[(df.index >= pd.Timestamp(START)) & (df.index <= pd.Timestamp(END) + pd.Timedelta(days=1))]
    buf = io.BytesIO()
    (df.to_frame() if isinstance(df, pd.Series) else df).to_csv(buf)
    buf.seek(0)
    st.download_button("⬇ CSV", buf, file_name=f"{label}.csv", mime="text/csv", key=f"dl_{label}_{id(df)}")

def render_two_col(charts):
    """Render (title, fig [, df]) tuples two per row (explicit unique keys - see usa_macro.py)."""
    n, i = len(charts), 0
    while i < n:
        if i == n - 1 and n % 2 != 0:
            item = charts[i]
            show_chart(item[1], key=f"chart_{item[0]}")
            if len(item) > 2 and item[2] is not None:
                csv_download(item[2], item[0])
            i += 1
        else:
            c1, c2 = st.columns(2)
            for col, item in [(c1, charts[i]), (c2, charts[i + 1])]:
                with col:
                    show_chart(item[1], key=f"chart_{item[0]}")
                    if len(item) > 2 and item[2] is not None:
                        csv_download(item[2], item[0])
            i += 2

def _z1y(series: pd.Series, window=252) -> np.ndarray:
    clean = series.dropna()
    z = (clean - clean.rolling(window).mean()) / clean.rolling(window).std()
    return z.reindex(series.index).values

def line_chart(df: pd.DataFrame, title, episodes, colors=None, ysuffix="", height=480, hline=None,
               z_window=None, dash=None, fmt=".2f", mode="lines"):
    fig = go.Figure()
    for i, c in enumerate(df.columns if df is not None else []):
        s = df[c].dropna()
        if s.empty:
            continue
        kw = {}
        if z_window:
            kw = dict(customdata=_z1y(s, z_window),
                      hovertemplate=f"%{{x|%Y-%m-%d}}<br>{c}: %{{y:{fmt}}}{ysuffix}<br>Z: %{{customdata:.2f}}<extra></extra>")
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=c, mode=mode,
                                 line=dict(color=(colors or {}).get(c) or PALETTE[i % len(PALETTE)],
                                           dash=(dash or {}).get(c)), **kw))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dot", line_color="#555")
    fig.update_layout(**base_layout(title, height))
    if ysuffix:
        fig.update_yaxes(ticksuffix=ysuffix)
    add_episodes(fig, episodes)
    return fig

def stacked_bars_with_line(parts: pd.DataFrame, total: pd.Series, title, episodes, total_name, colors=None,
                           yprefix="$", ysuffix="bn"):
    """Relative stacked bars of the components with the total as a line - the standard way to
    show a current-account (or any balance) decomposition."""
    fig = go.Figure()
    for i, c in enumerate(parts.columns):
        s = parts[c].dropna()
        fig.add_trace(go.Bar(x=s.index, y=s.values, name=c, marker_color=(colors or {}).get(c) or PALETTE[i % len(PALETTE)]))
    t = total.dropna()
    fig.add_trace(go.Scatter(x=t.index, y=t.values, name=total_name, mode="lines+markers",
                             line=dict(color="#e0e0e0", width=2), marker=dict(size=5)))
    fig.add_hline(y=0, line_dash="dot", line_color="#555")
    fig.update_layout(**base_layout(title))
    fig.update_layout(barmode="relative")
    fig.update_yaxes(tickprefix=yprefix, ticksuffix=ysuffix)
    add_episodes(fig, episodes)
    return fig

Z_SCORE_WINDOWS = {"1M": 21, "3M": 63, "1Y": 252}

def calculate_price_zscores(series: pd.Series, windows=None) -> dict:
    out = {}
    for label, window in (windows or Z_SCORE_WINDOWS).items():
        z = (series - series.rolling(window).mean()) / series.rolling(window).std()
        out[label] = float(z.iloc[-1]) if len(z) and pd.notna(z.iloc[-1]) else None
    return out

def calculate_period_returns(series: pd.Series, windows=None) -> dict:
    out = {}
    for label, window in (windows or Z_SCORE_WINDOWS).items():
        if len(series) <= window:
            out[label] = None
            continue
        past = series.iloc[-1 - window]
        out[label] = float((series.iloc[-1] / past - 1) * 100) if past else None
    return out

# ── Series catalogue (shared by the prefetch and the tabs, so cache keys match) ─
REQ_BOP  = ("BOP", "IDN.*.*.USD.Q", "2005")       # full BPM6 detail, USD, quarterly
REQ_IL   = ("IL", "IDN.*.USD.M", "2000")          # international reserves, monthly
REQ_CPI  = ("CPI", "IDN.CPI._T.*.M", "2000")      # headline CPI (index + YoY + MoM)
REQ_ITG  = ("ITG", "IDN.*.*.M", "2005")           # merchandise exports (FOB) / imports (CIF)
PARTNERS = {"CHN": "China", "USA": "United States", "JPN": "Japan", "IND": "India", "SGP": "Singapore",
            "MYS": "Malaysia", "KOR": "South Korea", "PHL": "Philippines", "VNM": "Vietnam", "THA": "Thailand",
            "AUS": "Australia", "NLD": "Netherlands", "SAU": "Saudi Arabia", "DEU": "Germany"}
REQ_IMTS = ("IMTS", "IDN.XG_FOB_USD+MG_CIF_USD." + "+".join(PARTNERS) + ".M", "2015")
REQ_IR   = ("MFS_IR", "IDN.*.M", "2005")          # BI policy-related, deposit, lending rates
REQ_ER   = ("ER", "IDN.XDC_USD.*.M", "2000")      # IDR per USD, monthly
REQ_MA   = ("MFS_MA", "IDN.*.*.M", "2005")        # broad / base money
REQ_QNEA = ("QNEA", "IDN.*.*.*.*.Q", "2005")      # GDP by expenditure
IMF_REQUESTS = [REQ_BOP, REQ_IL, REQ_CPI, REQ_ITG, REQ_IMTS, REQ_IR, REQ_ER, REQ_MA, REQ_QNEA]
BIS_REQUESTS = [("WS_CBPOL", "D.ID"), ("WS_EER", "M.R.B.ID"), ("WS_EER", "M.N.B.ID"),
                ("WS_SPP", "Q.ID.R.628"), ("WS_SPP", "Q.ID.N.628")]

IDX_TICKERS = {"^JKSE": "Jakarta Composite", "^JKLQ45": "LQ45", "EIDO": "EIDO (USD ETF)"}
STOCKS = {"BBCA.JK": "Bank Central Asia", "BBRI.JK": "Bank Rakyat Indonesia", "BMRI.JK": "Bank Mandiri",
          "BBNI.JK": "Bank Negara Indonesia", "TLKM.JK": "Telkom Indonesia", "ASII.JK": "Astra International",
          "ADRO.JK": "Adaro / AlamTri", "PTBA.JK": "Bukit Asam", "ANTM.JK": "Aneka Tambang", "INCO.JK": "Vale Indonesia",
          "UNVR.JK": "Unilever Indonesia", "ICBP.JK": "Indofood CBP", "AMMN.JK": "Amman Mineral", "GOTO.JK": "GoTo"}
MACRO_TICKERS = {"IDR=X": "USD/IDR", "BZ=F": "Brent", "TIO=F": "Iron Ore", "ALI=F": "Aluminium", "GC=F": "Gold"}
MARKETS_HIST_START = "2015-01-01"
VOL_TICKERS = {"^VIX": "VIX", "^MOVE": "MOVE"}  # ^MOVE = ICE BofA MOVE index (Yahoo's name label is wrong; data verified)

# ── Date range ────────────────────────────────────────────────────────────────
st.title("🇮🇩 Indonesia Macro Dashboard")
DATE_MIN = datetime.date(2005, 1, 1)
DATE_MAX = datetime.date.today()
st.session_state.setdefault("date_range_slider", (DATE_MAX.replace(year=DATE_MAX.year - 5), DATE_MAX))

def _sync_date_range_start():
    _, hi = st.session_state.date_range_slider
    st.session_state.date_range_slider = (st.session_state.date_range_start_input, hi)

def _sync_date_range_end():
    lo, _ = st.session_state.date_range_slider
    st.session_state.date_range_slider = (lo, st.session_state.date_range_end_input)

col_d1, col_d2, col_d3 = st.columns([3, 1, 1])
with col_d1:
    date_range = st.slider("Date Range", min_value=DATE_MIN, max_value=DATE_MAX, format="YYYY-MM-DD", key="date_range_slider")
with col_d2:
    st.date_input("Start (exact)", value=date_range[0], min_value=DATE_MIN, max_value=DATE_MAX,
                  key="date_range_start_input", on_change=_sync_date_range_start)
with col_d3:
    st.date_input("End (exact)", value=date_range[1], min_value=DATE_MIN, max_value=DATE_MAX,
                  key="date_range_end_input", on_change=_sync_date_range_end)
date_range = st.session_state.date_range_slider
START = date_range[0].strftime("%Y-%m-%d")
END = date_range[1].strftime("%Y-%m-%d")
episodes = episodes_in(START, END)

# ── Background prefetch ───────────────────────────────────────────────────────
for _req in IMF_REQUESTS:
    prefetch(_imf_raw, *_req)
for _flow, _key in BIS_REQUESTS:
    prefetch(_bis_raw, _flow, _key)
prefetch(_yf_batch_raw, tuple(IDX_TICKERS), MARKETS_HIST_START)
prefetch(_yf_batch_raw, tuple(STOCKS), MARKETS_HIST_START)
prefetch(_yf_batch_raw, tuple(MACRO_TICKERS), MARKETS_HIST_START)
start_comtrade_worker()
prefetch(_yf_batch_raw, tuple(VOL_TICKERS), MARKETS_HIST_START)
prefetch(_phei_curve_raw)
prefetch(_bi_inflation_raw)
prefetch(_bi_reserves_raw)
prefetch(_bis_raw, "WS_CBPOL", "D.US")
if FRED_API_KEY:
    for _sid in UST_SERIES.values():
        prefetch(_fred_raw, _sid)

# ── Shared derived series (full history; tabs slice) ─────────────────────────
def bop(raw, indicator, entry):
    return imf_pick(raw, INDICATOR=indicator, BOP_ACCOUNTING_ENTRY=entry) / 1e9   # USD bn

def gdp_real_nsa(raw):
    return imf_pick(raw, INDICATOR="B1GQ", PRICE_TYPE="Q", S_ADJUSTMENT="NSA", TYPE_OF_TRANSFORMATION="XDC")

# ── Summary bar ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Latest Readings & Z-Scores</div>', unsafe_allow_html=True)
Z_WINDOWS = {"D": {"1M": 21, "3M": 63, "12M": 252}, "M": {"3M": 3, "12M": 12, "36M": 36},
             "Q": {"4Q": 4, "8Q": 8, "20Q": 20}}

def _zscores(series, windows):
    out = {}
    series = series.dropna()
    for label, window in windows.items():
        if len(series) <= window + 1:
            out[label] = None
            continue
        z = (series - series.rolling(window).mean()) / series.rolling(window).std()
        out[label] = float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else None
    return out

@st.cache_data(ttl=3600)
def get_summary_metrics(end):
    """(value, delta, unit, delta_unit, z-scores, delta label, as-of) per card. Daily series
    compare to one calendar month earlier; monthly/quarterly to the previous print."""
    def load(req):
        return _imf_raw(*req)

    def cpi_yoy():
        return cpi_yoy_series(load(REQ_CPI))

    def bi_rate():
        return _bis_raw("WS_CBPOL", "D.ID")

    def real_rate():
        r = bi_rate().resample("MS").mean()
        return (r - cpi_yoy()).dropna()

    def gdp_yoy():
        g = gdp_real_nsa(load(REQ_QNEA))
        return (g.pct_change(4, fill_method=None) * 100).dropna()

    def ca():
        return bop(load(REQ_BOP), "CAB", "NETCD_T")

    def ca_4q():
        return ca().rolling(4).sum().dropna()

    def trade_bal():
        raw = load(REQ_ITG)
        return ((imf_pick(raw, INDICATOR="XG", TYPE_OF_TRANSFORMATION="FOB_USD") - imf_pick(raw, INDICATOR="MG", TYPE_OF_TRANSFORMATION="CIF_USD")) / 1e9).dropna()

    def reserves():
        return reserves_total_series(load(REQ_IL))

    def import_cover():
        m = imf_pick(load(REQ_ITG), INDICATOR="MG", TYPE_OF_TRANSFORMATION="CIF_USD").rolling(3).mean() / 1e9
        return (reserves() / m).dropna()

    def usdidr():
        df = _yf_batch_raw(tuple(MACRO_TICKERS), MARKETS_HIST_START)
        return df["IDR=X"]["Close"].dropna()

    def jci():
        df = _yf_batch_raw(tuple(IDX_TICKERS), MARKETS_HIST_START)
        return df["^JKSE"]["Close"].dropna()

    def m2_yoy():
        s = imf_pick(load(REQ_MA), INDICATOR="BM_MAI")
        return (s.pct_change(12, fill_method=None) * 100).dropna()

    specs = [
        ("CPI YoY",            cpi_yoy, "%", "M"),
        ("BI Rate",            bi_rate, "%", "D"),
        ("Real Policy Rate",   real_rate, "%", "M"),
        ("Real GDP YoY",       gdp_yoy, "%", "Q"),
        ("Current Account",    ca, "$bn", "Q"),
        ("CA, Last 4Q",        ca_4q, "$bn", "Q"),
        ("Trade Balance",      trade_bal, "$bn", "M"),
        ("FX Reserves",        reserves, "$bn", "M"),
        ("Import Cover",       import_cover, "mo", "M"),
        ("USD/IDR",            usdidr, "", "D"),
        ("Jakarta Composite",  jci, "", "D"),
        ("Broad Money YoY",    m2_yoy, "%", "M"),
    ]
    results = {}
    for name, loader, unit, freq in specs:
        try:
            s = loader().dropna()
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
            results[name] = (None, None, "", "", {}, "", "")
            continue
        if len(s) < 2:
            results[name] = (None, None, "", "", {}, "", "")
            continue
        latest = float(s.iloc[-1])
        if freq == "D":
            pv = s.asof(s.index[-1] - pd.DateOffset(months=1))
            prev = float(pv) if pd.notna(pv) else None
        else:
            prev = float(s.iloc[-2])
        delta = latest - prev if prev is not None else None
        delta_unit = unit
        if name == "BI Rate" and delta is not None:
            delta, delta_unit = delta * 100, "bps"
        label = {"D": "1M", "M": "MoM", "Q": "QoQ"}[freq]
        asof = s.index[-1]
        asof_s = asof.strftime("%d %b %Y") if freq == "D" else (f"Q{asof.quarter} {asof.year}" if freq == "Q" else asof.strftime("%b %Y"))
        results[name] = (latest, delta, unit, delta_unit, _zscores(s, Z_WINDOWS[freq]), label, asof_s)
    return results

def _fmt(v, unit):
    if unit == "$bn":
        return f"{'−' if v < 0 else ''}${abs(v):.1f}bn"
    if unit == "mo":
        return f"{v:.1f} mo"
    if unit == "bps":
        return f"{v:+.0f}bps"
    if unit == "":
        return f"{v:,.0f}"
    return f"{v:.2f}{unit}"

with st.spinner("Loading summary metrics…"):
    summary = get_summary_metrics(END)

summary_items = list(summary.items())
ROW_SIZE = 6
for row_start in range(0, len(summary_items), ROW_SIZE):
    cols = st.columns(ROW_SIZE)
    for col, (name, (val, delta, unit, delta_unit, z, dlabel, asof_s)) in zip(cols, summary_items[row_start:row_start + ROW_SIZE]):
        with col:
            if val is None:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{name}</div><div class="metric-value neutral">N/A</div></div>', unsafe_allow_html=True)
                continue
            val_str = _fmt(val, unit)
            if delta is None:
                delta_str = ""
            elif delta_unit in ("$bn",):
                delta_str = f"{delta:+.1f}bn"
            elif delta_unit == "mo":
                delta_str = f"{delta:+.1f} mo"
            elif delta_unit == "bps":
                delta_str = f"{delta:+.0f}bps"
            elif delta_unit == "":
                delta_str = f"{delta:+,.0f}"
            else:
                delta_str = f"{delta:+.2f}{delta_unit}"
            delta_cls = "positive" if (delta or 0) > 0 else "negative" if (delta or 0) < 0 else "neutral"
            z_spans = []
            for zlabel, zval in (z or {}).items():
                if zval is None:
                    z_spans.append(f'<span class="neutral">{zlabel} n/a</span>')
                else:
                    zcls = "positive" if zval > 0 else "negative" if zval < 0 else "neutral"
                    z_spans.append(f'<span class="{zcls}">{zlabel} {zval:+.1f}</span>')
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{name}</div>
              <div class="metric-value neutral">{val_str}</div>
              <div class="metric-delta {delta_cls}">{delta_str} {dlabel}</div>
              <div class="metric-z">{"".join(z_spans)}</div>
              <div style="font-size:0.58rem;color:#8a94a6;margin-top:3px">as of {asof_s}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── SBN Rates Monitor (outrights / spreads / flies) - port of usa_macro.py's Rates Monitor ──
# Uses the full recorded SBN history (not the date-slider slice), so z-scores and percentiles
# don't change when the date range narrows. Differences from the US version:
#   * funding / carry is the BI rate (BI's 7-day reverse repo policy rate), and
#   * the BI rate is the 0-year point of the curve, so the 1Y SBN "rolls down" to the BI rate -
#     which is what lets 1Y-based spreads and flies (e.g. 1s10s) get a carry & roll number,
#     whereas the US monitor leaves sub-1Y legs blank.
SBN_RM_TENORS = {"1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7, "10Y": 10, "15Y": 15, "20Y": 20, "30Y": 30}
# spread/fly = Σ weight × yield (bp). "Long" = steepener / long fly = pay the positive-weight legs.
SBN_RM_SPREADS = {"1s5s": {"5Y": 1, "1Y": -1}, "1s10s": {"10Y": 1, "1Y": -1}, "2s5s": {"5Y": 1, "2Y": -1},
                  "2s10s": {"10Y": 1, "2Y": -1}, "5s10s": {"10Y": 1, "5Y": -1}, "5s30s": {"30Y": 1, "5Y": -1},
                  "10s30s": {"30Y": 1, "10Y": -1}}
SBN_RM_FLIES = {"1s5s10s": {"5Y": 2, "1Y": -1, "10Y": -1}, "2s5s10s": {"5Y": 2, "2Y": -1, "10Y": -1},
                "5s10s30s": {"10Y": 2, "5Y": -1, "30Y": -1}, "10s20s30s": {"20Y": 2, "10Y": -1, "30Y": -1},
                "1s10s20s": {"10Y": 2, "1Y": -1, "20Y": -1}}

def _rm_pctile(s, years):
    w = s[s.index >= s.index[-1] - pd.DateOffset(years=years)]
    return float((w <= w.iloc[-1]).mean() * 100) if len(w) > 20 else np.nan

def _rm_chg(s, offset):
    prev = s.asof(s.index[-1] - offset)
    return float(s.iloc[-1] - prev) if pd.notna(prev) else np.nan

def build_sbn_monitor(sbn: pd.DataFrame, bi_rate: pd.Series):
    """Returns (outrights_df, spreads_df, flies_df, funding_level, funding_date)."""
    curve = sbn[[t for t in SBN_RM_TENORS if t in sbn]].dropna(how="all").sort_index()
    fund_s = bi_rate.dropna().sort_index()
    if curve.empty or fund_s.empty:
        raise ValueError("no recorded SBN curve or BI rate yet")
    fund_dt = curve.index[-1]
    fund = float(fund_s.asof(fund_dt))
    last = curve.iloc[-1].dropna()
    xs = np.array([0.0] + [float(SBN_RM_TENORS[t]) for t in last.index])   # BI rate = the 0Y point
    ys = np.array([fund] + list(last.values))
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]

    def _mdur(y, T):
        """Modified duration of a par bond (semi-annual coupons) with T years left; 0 once it has matured."""
        return 0.0 if T <= 0 else (1 - (1 + y / 200) ** (-2 * T)) / (y / 100)

    # Every number is expressed in bp of yield per unit of TODAY's DV01, so DV01-neutral legs simply add:
    #   breakeven carry = (yield − BI rate) / D(T)
    #   roll            = [y(T) − y(T−1)] × D(T−1) / D(T)
    # The roll uses the duration the bond will have a year from now - the yield drop from rolling down the
    # curve is earned on a bond that has become one year shorter. A 1Y held for a year matures at par, so its
    # whole excess return is carry and its roll is 0 (multiplying by the current duration double-counted it).
    legs = {}
    for t, T in SBN_RM_TENORS.items():
        if t not in last:
            continue
        y = float(last[t])
        y_h = float(np.interp(T - 1, xs, ys))                    # yield of the T-1 point on today's curve (0Y = BI rate)
        d_now, d_h = _mdur(y, T), _mdur(y_h, T - 1)
        running = (y - fund) * 100                               # bp/yr of notional, holding the tenor funded at BI rate
        be = running / d_now                                     # bp of yield (per unit of today's DV01)
        roll = (y - y_h) * 100 * d_h / d_now                     # bp of yield (per unit of today's DV01)
        legs[t] = (running, be, roll, y, d_now)

    def row(name, s, unit, running=np.nan, be=np.nan, roll=np.nan):
        s = s.dropna()
        m = 100 if unit == "%" else 1
        z = _zscores(s, Z_WINDOWS["D"])
        vol = s.diff().iloc[-252:].std() * m * np.sqrt(252) if len(s) > 21 else np.nan
        cr = be + roll
        return {"Instrument": name, "Level": float(s.iloc[-1]), "_unit": unit,
                "Δ1D (bp)": (s.iloc[-1] - s.iloc[-2]) * m if len(s) > 1 else np.nan,
                "Δ1W (bp)": _rm_chg(s, pd.Timedelta(days=7)) * m, "Δ1M (bp)": _rm_chg(s, pd.DateOffset(months=1)) * m,
                "Δ3M (bp)": _rm_chg(s, pd.DateOffset(months=3)) * m,
                "Z 1M": z["1M"], "Z 3M": z["3M"], "Z 12M": z["12M"], "1Y %ile": _rm_pctile(s, 1), "5Y %ile": _rm_pctile(s, 5),
                "Carry (bp/yr)": running, "Breakeven carry (bp yld/yr)": be, "Roll 1Y (bp)": roll,
                "Carry+Roll (bp yld/yr)": cr, "1Y vol (bp)": vol,
                "C+R / Vol": cr / vol if pd.notna(cr) and pd.notna(vol) and vol else np.nan, "As of": s.index[-1]}

    outr = [row("BI Rate", fund_s, "%")] + [row(f"{t} SBN", curve[t], "%", *legs[t][:3]) for t in SBN_RM_TENORS if t in legs]

    def combo(defs):
        rows = []
        for n, w in defs.items():
            if not all(k in legs for k in w):
                continue
            s_ = sum(curve[k] * v for k, v in w.items()).dropna() * 100
            be = -sum(v * legs[k][1] for k, v in w.items())       # DV01-neutral legs add in bp of yield
            roll = -sum(v * legs[k][2] for k, v in w.items())
            rows.append(row(n, s_, "bp", np.nan, be, roll))
        return rows
    front = ((curve["1Y"] - fund_s.reindex(curve.index, method="ffill")) * 100).dropna() if "1Y" in curve else pd.Series(dtype=float)
    spr = combo(SBN_RM_SPREADS) + ([row("1Y SBN − BI Rate", front, "bp")] if not front.empty else [])
    fly = combo(SBN_RM_FLIES)
    return pd.DataFrame(outr), pd.DataFrame(spr), pd.DataFrame(fly), fund, fund_dt, legs

def _rm_zcolor(v):
    if pd.isna(v):
        return "color: #5f6b7e"
    a = min(abs(v) / 2.5, 1.0)
    return f"background-color: rgba({'38,166,154' if v > 0 else '239,83,80'},{0.10 + 0.5 * a:.2f}); color: #e0e0e0"

def _rm_signcolor(v):
    return "color: #8a94a6" if pd.isna(v) or v == 0 else ("color: #26a69a" if v > 0 else "color: #ef5350")

def _rm_pctcolor(v):
    return "color: #5f6b7e" if pd.isna(v) else _rm_zcolor((v - 50) / 20)

def render_rates_monitor_table(df, key, show_running):
    """One monitor table with its own multi-column sort controls (priority order + per-column direction)."""
    if not show_running:
        df = df.drop(columns="Carry (bp/yr)")
    sortable = [c for c in df.columns if not c.startswith("_")]
    s1, s2 = st.columns([3, 2])
    sort_cols = s1.multiselect("Sort by (priority order — first pick sorts first)", sortable, default=[],
                               key=f"sbn_rm_sort_{key}", placeholder="Default order")
    asc = []
    if sort_cols:
        for dc, col in zip(s2.columns(len(sort_cols)), sort_cols):
            asc.append(dc.radio(col, ["↓ Desc", "↑ Asc"], key=f"sbn_rm_dir_{key}_{col}") == "↑ Asc")
    else:
        s2.caption("Pick one or more columns to sort by, or click a column header for a quick single-column sort.")
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=asc, na_position="last", kind="mergesort")
    view = df.copy()
    view["Level"] = [f"{v:.3f}%" if u == "%" else f"{v:+.1f}bp" for v, u in zip(view["Level"], view["_unit"])]
    view["As of"] = view["As of"].dt.strftime("%d %b")
    view = view.drop(columns="_unit")
    chg = ["Δ1D (bp)", "Δ1W (bp)", "Δ1M (bp)", "Δ3M (bp)"]
    carry = [c for c in ["Carry (bp/yr)", "Breakeven carry (bp yld/yr)", "Roll 1Y (bp)", "Carry+Roll (bp yld/yr)"] if c in view]
    # Display values are pre-formatted text ("—" for missing) - st.dataframe ignores a Styler's formatting for
    # missing values and prints "None" (verified 2026-10-09) - while the colours are computed from the numbers.
    formats = {**{c: "+.1f" for c in chg + carry}, "Z 1M": "+.2f", "Z 3M": "+.2f", "Z 12M": "+.2f",
               "1Y %ile": ".0f", "5Y %ile": ".0f", "1Y vol (bp)": ".0f", "C+R / Vol": "+.2f"}
    num = view.copy()
    for c, f in formats.items():
        if c in view:
            num[c] = pd.to_numeric(num[c], errors="coerce")
            view[c] = ["—" if pd.isna(v) else format(v, f) for v in num[c]]
    def _colors(fn, cols):
        return lambda col: [fn(v) for v in num[col.name]] if col.name in cols else [""] * len(col)
    zc = ["Z 1M", "Z 3M", "Z 12M", "C+R / Vol"]
    pc = ["1Y %ile", "5Y %ile"]
    styled = (view.style.apply(_colors(_rm_signcolor, chg + carry), subset=[c for c in chg + carry if c in view])
              .apply(_colors(_rm_zcolor, zc), subset=zc).apply(_colors(_rm_pctcolor, pc), subset=pc))
    st.dataframe(styled, hide_index=True, use_container_width=True, height=36 * (len(view) + 1))
    csv_download(df.drop(columns="_unit").set_index("Instrument"), f"sbn_rates_monitor_{key}")

tabs = st.tabs([
    "BI & Rates",
    "Prices",
    "Reserves & FX",
    "Current Account",
    "Trade",
    "Growth",
    "Indonesia Markets",
    "Economic Calendar",
])

WARMUP_START = (pd.Timestamp(START) - pd.DateOffset(months=ZSCORE_WARMUP_MONTHS)).strftime("%Y-%m-%d")

def _clip(df):
    """Chart-data clip: the selected range plus ~13 months of warm-up history for the z-score
    hover (finalize_fig trims the warm-up off before display)."""
    return _slice(df, WARMUP_START, END) if df is not None and len(df) else df

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — BI & Rates
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Bank Indonesia Policy & Rates")
    with st.spinner("Loading rates…"):
        bi_d = bis("WS_CBPOL", "D.ID", "BI Rate")
        ir_raw = imf_load(*REQ_IR, what="bank interest rates")
        cpi_raw = imf_load(*REQ_CPI, what="CPI")
        ma_raw = imf_load(*REQ_MA, what="money aggregates")
        mk = fetch_yf_batch(MACRO_TICKERS, MARKETS_HIST_START)

    with st.expander("Rates Monitor — SBN outrights, spreads & flies", expanded=True):
        with st.spinner("Building rates monitor…"):
            try:
                rm_out, rm_spr, rm_fly, rm_fund, rm_fund_dt, rm_legs = build_sbn_monitor(load_sbn_history(record=True), bi_d)
            except Exception as e:
                st.warning(f"Could not build the SBN rates monitor: {e}")
                rm_out = None
        if rm_out is not None:
            n_hist = len(load_sbn_history(record=False))
            st.caption(f"Carry is for holding each SBN tenor funded at the BI rate ({rm_fund:.2f}% on {rm_fund_dt:%d %b}), "
                       "and the BI rate is used as the 0-year point of the curve. Carry (bp/yr) = (yield − BI rate) × 100. "
                       "Breakeven carry = that ÷ modified duration. Roll 1Y = the yield drop from rolling one year down today's "
                       "PHEI curve × the bond's duration a year from now ÷ today's duration (so a 1Y, which simply matures, has "
                       "no roll - all its return is carry). Both are in bp of yield per unit of today's DV01, so DV01-neutral "
                       "spread/fly legs add straight up. C+R / Vol = (breakeven carry + roll) ÷ 1Y realised vol. "
                       "Changes, z-scores, percentiles and vol use the recorded PHEI history "
                       f"({n_hist} day{'s' if n_hist != 1 else ''} so far), so they fill in as it builds - carry & roll work from day one.")
            with st.expander("Outright Rates", expanded=True):
                render_rates_monitor_table(rm_out, "outright", show_running=True)
            rm_side = st.radio("Spreads & flies shown for", ["Steepener / long fly", "Flattener / short fly"], horizontal=True,
                               key="sbn_rm_side",
                               help="Steepener = receive the short leg, pay the long leg (profits if the spread widens). "
                                    "Flattener = the reverse. Long fly = pay the belly, receive the wings. Level, changes and "
                                    "z-scores describe the spread itself and don't change; carry and roll flip sign.")
            sign = 1 if rm_side.startswith("Steepener") else -1
            pnl_cols = ["Breakeven carry (bp yld/yr)", "Roll 1Y (bp)", "Carry+Roll (bp yld/yr)", "C+R / Vol"]

            def _side(df):
                df = df.copy()
                for c in pnl_cols:
                    if c in df:
                        df[c] = df[c] * sign
                return df
            with st.expander("Spreads", expanded=True):
                render_rates_monitor_table(_side(rm_spr), "spreads", show_running=False)
            with st.expander("Flies", expanded=True):
                render_rates_monitor_table(_side(rm_fly), "flies", show_running=False)

            with st.expander("Leg breakdown — how a spread's carry & roll nets out", expanded=False):
                combos = {**SBN_RM_SPREADS, **SBN_RM_FLIES}
                pick = st.selectbox("Spread or fly", [n for n, w in combos.items() if all(k in rm_legs for k in w)],
                                    index=list(combos).index("1s10s") if "1s10s" in combos else 0, key="sbn_rm_leg_pick")
                w = combos[pick]
                # DV01-neutral notionals: each leg's |weight| units of DV01, i.e. notional ∝ |weight| / duration,
                # scaled so the longest-tenor leg is 1.00
                ref = max(w, key=lambda k: SBN_RM_TENORS[k])
                rows = []
                for k, wt in sorted(w.items(), key=lambda kv: SBN_RM_TENORS[kv[0]]):
                    running, be, roll, y, d_now = rm_legs[k]
                    pos = -1 * np.sign(wt) * sign                          # +1 = receive (long the bond), -1 = pay
                    rows.append({"Leg": f"{k} SBN", "Yield": f"{y:.3f}%", "Position": "Receive (long bond)" if pos > 0 else "Pay (short bond)",
                                 "DV01 weight": abs(wt), "Notional vs longest leg": abs(wt) / d_now / (abs(w[ref]) / rm_legs[ref][4]),
                                 "Carry (bp yld/yr)": pos * abs(wt) * be, "Roll (bp)": pos * abs(wt) * roll,
                                 "Carry+Roll (bp)": pos * abs(wt) * (be + roll)})
                legs_df = pd.DataFrame(rows)
                total = {"Leg": "Net", "Yield": "", "Position": rm_side, "DV01 weight": np.nan, "Notional vs longest leg": np.nan,
                         **{c: legs_df[c].sum() for c in ["Carry (bp yld/yr)", "Roll (bp)", "Carry+Roll (bp)"]}}
                legs_df = pd.concat([legs_df, pd.DataFrame([total])], ignore_index=True)
                disp = legs_df.copy()
                for c, f in [("DV01 weight", ".0f"), ("Notional vs longest leg", ".2f"), ("Carry (bp yld/yr)", "+.1f"),
                             ("Roll (bp)", "+.1f"), ("Carry+Roll (bp)", "+.1f")]:
                    disp[c] = ["" if pd.isna(v) else format(v, f) for v in legs_df[c]]
                st.table(disp.set_index("Leg"))
                st.caption("Each leg is sized to its DV01 weight (spreads 1:1, flies 1:2:1), so a parallel move nets to zero - "
                           "which needs a much larger notional in the short leg. Carry and roll are in bp of yield per unit of "
                           "DV01, so the legs add straight up to the net. Short tenors carry more per unit of DV01 (their "
                           "yield-minus-BI-rate is spread over a small duration), which is why steepeners earn and "
                           "flatteners pay carry on a steep front end.")

    deposit = imf_pick(ir_raw, INDICATOR="MFS135_RT_PT_A_PT").rename("Deposit Rate")
    lending = imf_pick(ir_raw, INDICATOR="MFS162_RT_PT_A_PT").rename("Lending Rate")
    fx_dep = imf_pick(ir_raw, INDICATOR="MFS135_FC_RT_PT_A_PT").rename("USD Deposit Rate")
    cpi_yoy = cpi_yoy_series(cpi_raw).rename("CPI YoY")

    fig_r = go.Figure()
    s = _clip(bi_d)
    if not s.empty:
        fig_r.add_trace(go.Scatter(x=s.index, y=s.values, name="BI Rate (policy)", line=dict(color="#ef5350", shape="hv", width=2.2)))
    for ser, color in [(lending, "#ff9800"), (deposit, "#42a5f5"), (fx_dep, "#8a94a6")]:
        c = _clip(ser).dropna()
        if not c.empty:
            fig_r.add_trace(go.Scatter(x=c.index, y=c.values, name=ser.name, line=dict(color=color)))
    c = _clip(cpi_yoy).dropna()
    if not c.empty:
        fig_r.add_trace(go.Scatter(x=c.index, y=c.values, name="CPI YoY", line=dict(color="#26a69a", dash="dot")))
    fig_r.update_layout(**base_layout("BI Rate vs Bank Rates & Inflation"))
    fig_r.update_yaxes(ticksuffix="%")
    add_episodes(fig_r, episodes)

    real = pd.DataFrame()
    if not bi_d.empty and not cpi_yoy.empty:
        real = pd.concat([bi_d.resample("MS").mean().rename("BI Rate (monthly avg)"), cpi_yoy], axis=1).dropna()
        real["Real Policy Rate"] = real["BI Rate (monthly avg)"] - real["CPI YoY"]
        real = _clip(real).round(3)
    fig_real = go.Figure()
    if not real.empty:
        fig_real.add_trace(go.Bar(x=real.index, y=real["Real Policy Rate"], name="Real policy rate (BI rate − CPI YoY)",
                                  marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in real["Real Policy Rate"]]))
    spread = _clip((lending - deposit).dropna().rename("Lending − Deposit Spread"))
    if spread is not None and not spread.empty:
        fig_real.add_trace(go.Scatter(x=spread.index, y=spread.values, name="Lending − deposit spread", yaxis="y2",
                                      line=dict(color="#ff9800")))
    fig_real.update_layout(**dual_axis_layout("Real Policy Rate & Bank Lending Spread", "Real rate (%)", "Spread (pp)"))
    add_episodes(fig_real, episodes)

    money = pd.DataFrame({
        "Broad Money (M2) YoY %": imf_pick(ma_raw, INDICATOR="BM_MAI").pct_change(12, fill_method=None) * 100,
        "Base Money YoY %": imf_pick(ma_raw, INDICATOR="NDMBM_MAI").pct_change(12, fill_method=None) * 100,
        "Currency in Circulation YoY %": imf_pick(ma_raw, INDICATOR="CIC_OUTDCORP_MAI").pct_change(12, fill_method=None) * 100,
    })
    money = _clip(money.dropna(how="all")).round(2)
    fig_money = line_chart(money, "Money Growth (YoY %)", episodes, ysuffix="%", hline=0, fmt=".1f")

    fig_bifx = go.Figure()
    idr_full = mk["IDR=X"]
    if not idr_full.empty:
        c = _clip(idr_full)
        fig_bifx.add_trace(go.Scatter(x=c.index, y=c["USD/IDR"], name="USD/IDR", line=dict(color="#42a5f5")))
    s = _clip(bi_d)
    if not s.empty:
        fig_bifx.add_trace(go.Scatter(x=s.index, y=s.values, name="BI Rate (rhs)", yaxis="y2", line=dict(color="#ef5350", shape="hv")))
    fig_bifx.update_layout(**dual_axis_layout("Rupiah vs BI Rate — Rate Moves to Defend the Currency", "IDR per USD", "BI Rate (%)"))
    add_episodes(fig_bifx, episodes)

    render_two_col([
        ("BI Rate vs Bank Rates", fig_r, _clip(pd.concat([bi_d, lending, deposit, fx_dep, cpi_yoy], axis=1))),
        ("Real Policy Rate", fig_real, real),
        ("Money Growth", fig_money, money),
        ("Rupiah vs BI Rate", fig_bifx, None),
    ])
    st.caption("BI Rate: daily from the BIS central-bank policy rate database. Deposit/lending rates: IMF Monetary & "
               "Financial Statistics (monthly, ~2-month lag).")

    # ── SBN curve & IDR-vs-US rate beta ─────────────────────────────────────────
    st.markdown('<div class="section-header">Government Bond Curve (SBN) & Beta of IDR Rates to US Rates</div>',
                unsafe_allow_html=True)
    with st.spinner("Loading SBN curve & US rates…"):
        sbn = load_sbn_history(record=True)
        fed_d = bis("WS_CBPOL", "D.US", "Fed Policy Rate")
        ust = {}
        for tenor, sid in UST_SERIES.items():
            try:
                ust[tenor] = _fred_raw(sid)
            except Exception as e:
                st.warning(f"Could not load US Treasury {tenor} ({sid}) from FRED: {e}")

    def _beta_stats(y, x):
        d = pd.concat([y, x], axis=1).dropna()
        if len(d) < 3 or d.iloc[:, 1].var() == 0:
            return None
        beta = d.cov().iloc[0, 1] / d.iloc[:, 1].var()
        corr = d.corr().iloc[0, 1]
        return {"Beta": round(beta, 2), "Correlation": round(corr, 2), "R²": round(corr ** 2, 2), "Obs": len(d)}

    # ── SBN curve snapshots (same pattern as the Fed funds futures section in usa_macro.py) ──
    # As-of slider + date picker over every recorded day; the selected date's curve is drawn over
    # the latest curve (ghosted), with fixed 1W/1M/3M/1Y-ago snapshot lines and a full-history
    # chart marked at the slider date.
    SBN_SNAPSHOT_COLORS = {"1W Ago": "#ff9800", "1M Ago": "#66bb6a", "3M Ago": "#e040fb", "1Y Ago": "#ef5350"}
    fig_asof, fig_snap, fig_hist = go.Figure(), go.Figure(), go.Figure()
    asof_tbl, snap_tbl, hist_tbl = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    as_of = None
    if not sbn.empty:
        tenors = list(sbn.columns)
        xs = [float(c[:-1]) for c in tenors]
        dates = list(sbn.index)
        last_d = dates[-1]
        today_row = sbn.loc[last_d, tenors]

        if len(dates) >= 2:
            if st.session_state.get("sbn_as_of_slider") not in dates:
                st.session_state["sbn_as_of_slider"] = last_d

            def _sync_sbn_from_picker():
                picked = pd.Timestamp(st.session_state.sbn_as_of_picker)
                idx = sbn.index.get_indexer([picked], method="nearest")[0]
                st.session_state.sbn_as_of_slider = sbn.index[idx]

            # The picker mirrors the slider through session state rather than value= - a value= that changes
            # every run makes Streamlit treat the picker as a new widget and drop the user's pick.
            st.session_state["sbn_as_of_picker"] = st.session_state["sbn_as_of_slider"].date()
            col_sl, col_dt = st.columns([3, 1])
            with col_sl:
                st.select_slider("SBN curve as of (for snapshot comparison)", options=dates,
                                 format_func=lambda d: d.strftime("%b %d, %Y"), key="sbn_as_of_slider")
            with col_dt:
                st.date_input("Or pick a date", min_value=dates[0].date(), max_value=last_d.date(),
                              key="sbn_as_of_picker", on_change=_sync_sbn_from_picker)
            as_of = st.session_state.sbn_as_of_slider
        else:
            as_of = last_d
            st.caption("Only one recorded day so far - the as-of slider appears once a second day is recorded.")
        asof_row = sbn.loc[as_of, tenors]
        as_of_label = as_of.strftime("%b %d, %Y")
        chg_bps = (today_row - asof_row) * 100

        # 1) As-of curve vs latest (ghosted), with per-tenor change bars on the right axis
        fig_asof.add_trace(go.Bar(x=xs, y=chg_bps.values, name=f"Change {as_of:%d %b} → {last_d:%d %b} (bps, rhs)", yaxis="y2",
                                  marker_color=["#ef5350" if v > 0 else "#26a69a" if v < 0 else "#8a94a6" for v in chg_bps.values],
                                  opacity=0.45, hovertemplate="%{x}Y: %{y:+.1f}bps<extra></extra>"))
        fig_asof.add_trace(go.Scatter(x=xs, y=today_row.values, mode="lines+markers", name=f"Latest ({last_d:%b %d, %Y})",
                                      line=dict(color="rgba(230,233,240,0.35)", width=2), hovertemplate="%{x}Y: %{y:.3f}%<extra></extra>"))
        fig_asof.add_trace(go.Scatter(x=xs, y=asof_row.values, mode="lines+markers", name=f"As of {as_of_label}",
                                      line=dict(color="#ffffff", width=2.4), fill="tonexty", fillcolor="rgba(255,255,255,0.08)",
                                      hovertemplate="%{x}Y: %{y:.3f}%<extra></extra>"))
        fig_asof.update_layout(**dual_axis_layout(f"SBN Yield Curve — As Of {as_of_label}", "Yield (%)", "Change to latest (bps)"))
        fig_asof.update_xaxes(title="Tenor (years)")
        asof_tbl = pd.DataFrame({"Tenor": tenors, f"Latest {last_d.date()}": today_row.values,
                                 f"As of {as_of.date()}": asof_row.values, "Change (bps)": chg_bps.round(1).values})
        ust_latest = {t: (ust[t].asof(last_d) if t in ust and not ust[t].empty else np.nan) for t in tenors}
        asof_tbl["UST latest"] = [ust_latest[t] for t in tenors]
        asof_tbl["SBN − UST (bps)"] = ((asof_tbl[f"Latest {last_d.date()}"] - asof_tbl["UST latest"]) * 100).round(1)

        # 2) Snapshots: latest + as-of + fixed look-backs (each only once that much history is recorded)
        fig_snap.add_trace(go.Scatter(x=xs, y=today_row.values, mode="lines+markers", name=f"Latest ({last_d:%b %d})",
                                      line=dict(color="#4fc3f7", width=2)))
        if as_of != last_d:
            fig_snap.add_trace(go.Scatter(x=xs, y=asof_row.values, mode="lines+markers", name=f"As of {as_of_label}",
                                          line=dict(color="#ffffff", width=2.2), fill="tonexty", fillcolor="rgba(255,255,255,0.08)"))
        snap_tbl = pd.DataFrame({"Tenor": tenors, f"Latest {last_d.date()}": today_row.values})
        missing_snaps = []
        for label, offset in [("1W Ago", pd.Timedelta(days=7)), ("1M Ago", pd.DateOffset(months=1)),
                              ("3M Ago", pd.DateOffset(months=3)), ("1Y Ago", pd.DateOffset(years=1))]:
            sub = sbn.loc[:last_d - offset]
            if sub.empty:
                missing_snaps.append(label)
                continue
            d, row = sub.index[-1], sub.iloc[-1][tenors]
            fig_snap.add_trace(go.Scatter(x=xs, y=row.values, mode="lines+markers", name=f"{label} ({d:%b %d})",
                                          line=dict(color=SBN_SNAPSHOT_COLORS[label], width=1, dash="dash")))
            snap_tbl[f"{label} {d.date()}"] = row.values
        fig_snap.update_layout(**base_layout("SBN Yield Curve — Snapshots"))
        fig_snap.update_xaxes(title="Tenor (years)")
        fig_snap.update_yaxes(ticksuffix="%")

        # 3) Full recorded history of key tenors, marker at the slider date
        hist_cols = [t for t in ["1Y", "5Y", "10Y", "30Y"] if t in sbn]
        hist_tbl = sbn[hist_cols]
        for t, color in zip(hist_cols, ["#42a5f5", "#26a69a", "#ff9800", "#ab47bc"]):
            fig_hist.add_trace(go.Scatter(x=sbn.index, y=sbn[t], mode="lines+markers", name=f"SBN {t}", line=dict(color=color, width=1.6)))
        fig_hist.add_vline(x=as_of, line_dash="solid", line_color="#4fc3f7", line_width=1.5)
        fig_hist.update_layout(**base_layout("SBN Yields — Recorded History (marker = slider date)"))
        fig_hist.update_yaxes(ticksuffix="%")

    # Policy-rate beta: BI rate vs Fed (available now, full history)
    fig_pol = go.Figure()
    pol_tbl = pd.DataFrame()
    if not bi_d.empty and not fed_d.empty:
        both = pd.concat([bi_d, fed_d], axis=1).ffill().dropna()
        c = _clip(both)
        fig_pol.add_trace(go.Scatter(x=c.index, y=c["BI Rate"], name="BI Rate", line=dict(color="#ef5350", shape="hv")))
        fig_pol.add_trace(go.Scatter(x=c.index, y=c["Fed Policy Rate"], name="Fed policy rate", line=dict(color="#42a5f5", shape="hv")))
        fig_pol.add_trace(go.Scatter(x=c.index, y=c["BI Rate"] - c["Fed Policy Rate"], name="BI − Fed spread (rhs)", yaxis="y2",
                                     line=dict(color="#8a94a6", dash="dot")))
        q = both.resample("QE").last().diff().dropna()
        rows = []
        for label, since in [("Since 2005", "2005-01-01"), ("Last 10 years", (pd.Timestamp.today() - pd.DateOffset(years=10)).strftime("%Y-%m-%d")),
                             ("Last 5 years", (pd.Timestamp.today() - pd.DateOffset(years=5)).strftime("%Y-%m-%d"))]:
            st_ = _beta_stats(q.loc[since:, "BI Rate"], q.loc[since:, "Fed Policy Rate"])
            if st_:
                rows.append({"Window": label, **st_})
        pol_tbl = pd.DataFrame(rows)
    fig_pol.update_layout(**dual_axis_layout("BI Rate vs Fed Policy Rate", "%", "Spread (pp)"))
    add_episodes(fig_pol, episodes)
    render_two_col([
        ("SBN Yield Curve As Of", fig_asof, asof_tbl),
        ("SBN Yield Curve Snapshots", fig_snap, snap_tbl),
        ("SBN Yields Recorded History", fig_hist, hist_tbl),
        ("BI Rate vs Fed", fig_pol, pol_tbl),
    ])
    if not sbn.empty and missing_snaps:
        st.caption(f"Snapshot lines {', '.join(missing_snaps)} appear automatically once that much SBN history has been "
                   f"recorded (recording started {sbn.index.min():%b %d, %Y}).")
    if not pol_tbl.empty:
        st.markdown("**Policy-rate beta** — quarterly change in the BI rate regressed on the quarterly change in the Fed "
                    "policy rate (beta 0.5 = BI typically moves 50bp for every 100bp the Fed moves):")
        st.table(pol_tbl.set_index("Window").style.format({"Beta": "{:.2f}", "Correlation": "{:.2f}", "R²": "{:.2f}"}))

    # Market-rate beta: SBN vs UST (needs recorded history)
    MIN_BETA_OBS = 20
    beta_win = int(st.number_input("SBN-vs-UST beta window (recorded trading days)", min_value=10, max_value=2500, value=60,
                                   step=5, key="idn_beta_window"))
    beta_rows, roll_frames = [], {}
    for tenor in UST_SERIES:
        if tenor not in sbn or tenor not in ust:
            continue
        y = sbn[tenor].dropna()
        # Jakarta closes before New York opens, so each Jakarta day is matched to the US close of the PREVIOUS
        # business day; changes are taken between consecutive recorded dates on both sides (same intervals).
        ux = ust[tenor].reindex(ust[tenor].index.union(y.index - pd.offsets.BDay(1))).ffill().reindex(y.index - pd.offsets.BDay(1))
        ux.index = y.index
        dy, dx = (y.diff() * 100).tail(beta_win), (ux.diff() * 100).tail(beta_win)
        st_ = _beta_stats(dy, dx)
        n = int(dy.notna().sum())
        beta_rows.append({"Pair": f"SBN {tenor} vs UST {tenor} (US lagged 1 day)",
                          **(st_ if st_ and n >= MIN_BETA_OBS else {"Beta": None, "Correlation": None, "R²": None, "Obs": n})})
        if n >= beta_win + 5:
            d = pd.concat([dy.rename("y"), dx.rename("x")], axis=1).dropna()
            roll_frames[f"{tenor} beta"] = d["y"].rolling(beta_win).cov(d["x"]) / d["x"].rolling(beta_win).var()
    beta_tbl = pd.DataFrame(beta_rows)
    n_rec = len(sbn)
    if n_rec - 1 < MIN_BETA_OBS:
        st.info(f"SBN market-rate beta needs at least {MIN_BETA_OBS} daily changes - {max(n_rec - 1, 0)} recorded so far "
                f"(recording started {sbn.index.min().date() if n_rec else 'today'}; one new day each trading day the dashboard "
                f"is opened). The policy-rate beta above uses full history and is available now.")
    if not beta_tbl.empty:
        st.markdown("**Market-rate beta** — daily change in SBN yield (bps) regressed on the prior-day change in the "
                    "matching US Treasury yield:")
        st.table(beta_tbl.set_index("Pair").style.format({"Beta": "{:.2f}", "Correlation": "{:.2f}", "R²": "{:.2f}"}, na_rep="—"))
    if roll_frames:
        show_chart(line_chart(_clip(pd.DataFrame(roll_frames).dropna(how="all")), f"Rolling {beta_win}-Day Beta of SBN Yields to UST Yields",
                              episodes, hline=1, fmt=".2f"), key="chart_idn_rolling_beta")
    st.caption("SBN curve: PHEI (Indonesia Bond Pricing Agency) IGSYC fitted curve, recorded daily by this dashboard into "
               "data/idr_sbn_yields.csv since 2026-10-07 - no free source offers historical SBN yields. US yields: FRED "
               "(DGS1/DGS5/DGS10). Fed policy rate: BIS.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Prices
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Inflation & Prices")
    with st.spinner("Loading prices…"):
        cpi_raw = imf_load(*REQ_CPI, what="CPI")
        er_raw = imf_load(*REQ_ER, what="exchange rates")
        spp_r = bis("WS_SPP", "Q.ID.R.628", "Real House Prices", warn=False)
        spp_n = bis("WS_SPP", "Q.ID.N.628", "Nominal House Prices")

    cpi_idx = imf_pick(cpi_raw, TYPE_OF_TRANSFORMATION="IX").rename("CPI Index")
    cpi_yoy = cpi_yoy_series(cpi_raw).rename("CPI YoY %")
    cpi_mom = imf_pick(cpi_raw, TYPE_OF_TRANSFORMATION="POP_PCH_PA_PT").rename("CPI MoM %")

    fig_cpi = go.Figure()
    m = _clip(cpi_mom).dropna()
    if not m.empty:
        fig_cpi.add_trace(go.Bar(x=m.index, y=m.values, name="MoM % (NSA)", yaxis="y2", opacity=0.45,
                                 marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in m.values]))
    y = _clip(cpi_yoy).dropna()
    if not y.empty:
        fig_cpi.add_trace(go.Scatter(x=y.index, y=y.values, name="CPI YoY %", line=dict(color="#42a5f5", width=2.2)))
        ma6 = _clip(cpi_yoy.rolling(6).mean()).dropna()
        fig_cpi.add_trace(go.Scatter(x=ma6.index, y=ma6.values, name="YoY 6M MA", line=dict(color="#42a5f5", dash="dot", width=1.2)))
    band_start = max(pd.Timestamp("2024-01-01"), pd.Timestamp(START))
    if band_start <= pd.Timestamp(END):
        fig_cpi.add_shape(type="rect", x0=band_start, x1=END, y0=1.5, y1=3.5, yref="y", fillcolor="rgba(38,166,154,0.10)", line_width=0)
        fig_cpi.add_annotation(x=band_start, y=3.5, text="BI target 2.5% ±1", showarrow=False, xanchor="left", yanchor="bottom",
                               font=dict(size=10, color="#26a69a"))
    fig_cpi.update_layout(**dual_axis_layout("Headline CPI — YoY & MoM", "YoY %", "MoM %"))
    add_episodes(fig_cpi, episodes)

    idr_m = imf_pick(er_raw, INDICATOR="XDC_USD", TYPE_OF_TRANSFORMATION="PA_RT").rename("USD/IDR (monthly avg)")
    pt = pd.concat([cpi_yoy, (idr_m.pct_change(12, fill_method=None) * 100).rename("Rupiah Depreciation YoY %")], axis=1).dropna()
    pt = _clip(pt).round(2)
    fig_pt = go.Figure()
    if not pt.empty:
        fig_pt.add_trace(go.Bar(x=pt.index, y=pt["Rupiah Depreciation YoY %"], name="USD/IDR YoY % (+ = weaker rupiah)",
                                marker_color="rgba(239,83,80,0.45)", yaxis="y2"))
        fig_pt.add_trace(go.Scatter(x=pt.index, y=pt["CPI YoY %"], name="CPI YoY %", line=dict(color="#42a5f5", width=2)))
    fig_pt.update_layout(**dual_axis_layout("Exchange-Rate Pass-Through — CPI vs Rupiah Weakness", "CPI YoY %", "USD/IDR YoY %"))
    add_episodes(fig_pt, episodes)

    hp = pd.DataFrame({"Nominal House Prices YoY %": spp_n.pct_change(4, fill_method=None) * 100,
                       "Real House Prices YoY %": spp_r.pct_change(4, fill_method=None) * 100})
    hp = _clip(hp.dropna(how="all")).round(2)
    fig_hp = line_chart(hp, "Residential Property Prices (YoY %, BIS)", episodes, ysuffix="%", hline=0, fmt=".1f", mode="lines+markers")

    lvl = _clip(cpi_idx).dropna()
    fig_lvl = go.Figure()
    if not lvl.empty:
        ann = ((cpi_idx / cpi_idx.shift(3)) ** 4 - 1) * 100
        a = _clip(ann).dropna()
        fig_lvl.add_trace(go.Scatter(x=lvl.index, y=lvl.values, name="CPI Index", line=dict(color="#90a4d4")))
        fig_lvl.add_trace(go.Scatter(x=a.index, y=a.values, name="3M annualised % (NSA)", yaxis="y2", line=dict(color="#ff9800", width=1.2)))
    fig_lvl.update_layout(**dual_axis_layout("CPI Level & 3-Month Annualised Momentum", "Index", "% annualised"))
    add_episodes(fig_lvl, episodes)

    render_two_col([
        ("Headline CPI", fig_cpi, _clip(pd.concat([cpi_yoy, cpi_mom], axis=1))),
        ("Exchange Rate Pass-Through", fig_pt, pt),
        ("CPI Momentum", fig_lvl, _clip(cpi_idx)),
        ("Property Prices", fig_hp, hp),
    ])

    st.markdown('<div class="section-header">Inflation Breakdown — Core / Administered / Volatile Food & Expenditure Groups</div>',
                unsafe_allow_html=True)
    if not BPS_API_KEY:
        st.info("Indonesia's inflation breakdown (core, administered prices, volatile food, and the 11 expenditure "
                "groups) is only published by BPS (Statistics Indonesia), whose API needs a free key: register at "
                "webapi.bps.go.id/developer, then add `BPS_API_KEY=...` to your .env - same pattern as FRED_API_KEY. "
                "The IMF's CPI-by-category series for Indonesia stopped updating after BPS rebased the CPI in 2025, so "
                "there's no free keyless source for it.")
    else:
        st.info("BPS_API_KEY found - the BPS breakdown loader still needs wiring against BPS's variable IDs "
                "(they can only be looked up with a key). Ask Claude to finish this section.")
    st.caption("Headline CPI: IMF CPI dataset (BPS source data), extended with the newest month(s) from Bank "
               "Indonesia's inflation page (YoY only; identical to the IMF on every overlapping month). The very latest "
               "BPS release can still be a month ahead of both - that needs BPS_API_KEY. MoM is not seasonally adjusted, so "
               "Ramadan/Eid months show seasonal food & transport spikes. BI's inflation target band shown is the "
               "2.5% ±1pp target in force since 2024.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Reserves & FX
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Foreign Exchange Reserves & the Rupiah")
    with st.spinner("Loading reserves…"):
        il_raw = imf_load(*REQ_IL, what="international reserves")
        itg_raw = imf_load(*REQ_ITG, what="merchandise trade")
        mk = fetch_yf_batch(MACRO_TICKERS, MARKETS_HIST_START)
        reer = bis("WS_EER", "M.R.B.ID", "REER (real, broad)")
        neer = bis("WS_EER", "M.N.B.ID", "NEER (nominal, broad)")

    comp = pd.DataFrame({
        "Foreign Exchange": imf_pick(il_raw, INDICATOR="RXF11FX_REVS"),
        "Gold": imf_pick(il_raw, INDICATOR="RGOLDNV_REVS"),
        "SDRs": imf_pick(il_raw, INDICATOR="RXDR_REVS"),
        "IMF Reserve Position": imf_pick(il_raw, INDICATOR="TRRPIMF_REVS"),
    }) / 1e9
    total_res = reserves_total_series(il_raw)
    comp_c, total_c = _clip(comp.dropna(how="all")), _clip(total_res)
    fig_res = go.Figure()
    for c_, color in zip(comp_c.columns, ["#42a5f5", "#eda100", "#ab47bc", "#26a69a"]):
        fig_res.add_trace(go.Bar(x=comp_c.index, y=comp_c[c_], name=c_, marker_color=color))
    if total_c is not None and not total_c.empty:
        fig_res.add_trace(go.Scatter(x=total_c.index, y=total_c.values, name="Total reserves", line=dict(color="#e0e0e0", width=2)))
    fig_res.update_layout(**base_layout("Official Reserve Assets by Type ($bn, Monthly)"))
    fig_res.update_layout(barmode="stack")
    fig_res.update_yaxes(tickprefix="$", ticksuffix="bn")
    add_episodes(fig_res, episodes)

    imports_m = imf_pick(itg_raw, INDICATOR="MG", TYPE_OF_TRANSFORMATION="CIF_USD") / 1e9
    cover = _clip((total_res / imports_m.rolling(3).mean()).dropna().rename("Months of Imports")).round(2)
    fig_cover = go.Figure()
    if cover is not None and not cover.empty:
        fig_cover.add_trace(go.Scatter(x=cover.index, y=cover.values, name="Reserves ÷ avg monthly imports (3M)",
                                       line=dict(color="#26a69a"), fill="tozeroy", fillcolor="rgba(38,166,154,0.10)"))
    fig_cover.add_hline(y=3, line_dash="dash", line_color="#ef5350", annotation_text="3-month IMF rule of thumb")
    fig_cover.update_layout(**base_layout("Import Cover (Months of Goods Imports)"))
    add_episodes(fig_cover, episodes)

    idr_full = mk["IDR=X"]
    dres = total_res.diff()
    idr_mo = (idr_full["USD/IDR"].resample("MS").last().pct_change(fill_method=None) * 100) if not idr_full.empty else pd.Series(dtype=float)
    intv = _clip(pd.concat([dres.rename("Reserves Δ ($bn)"), idr_mo.rename("USD/IDR % change")], axis=1).dropna()).round(3)
    fig_intv = go.Figure()
    if not intv.empty:
        fig_intv.add_trace(go.Bar(x=intv.index, y=intv["Reserves Δ ($bn)"], name="Monthly change in reserves ($bn)",
                                  marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in intv["Reserves Δ ($bn)"]]))
        fig_intv.add_trace(go.Scatter(x=intv.index, y=intv["USD/IDR % change"], name="USD/IDR % change (+ = weaker rupiah)",
                                      yaxis="y2", line=dict(color="#ff9800")))
    fig_intv.update_layout(**dual_axis_layout("FX Intervention Gauge — Reserve Changes vs Rupiah Moves", "$bn", "%"))
    add_episodes(fig_intv, episodes)

    fig_idr = go.Figure()
    if not idr_full.empty:
        c = _clip(idr_full)
        fig_idr.add_trace(go.Scatter(x=c.index, y=c["USD/IDR"], name="USD/IDR", line=dict(color="#42a5f5"),
                                     customdata=pd.Series(_z1y(idr_full["USD/IDR"]), index=idr_full.index).reindex(c.index).values,
                                     hovertemplate="%{x|%Y-%m-%d}<br>USD/IDR: %{y:,.0f}<br>Z (1Y): %{customdata:.2f}<extra></extra>"))
        add_ma_overlays(fig_idr, idr_full, "USD/IDR", START, END)
    fig_idr.update_layout(**base_layout("USD/IDR (Daily, higher = weaker rupiah)"))
    add_episodes(fig_idr, episodes)

    eer = _clip(pd.concat([reer, neer], axis=1).dropna(how="all"))
    fig_eer = line_chart(eer, "Rupiah Effective Exchange Rates (BIS, 2020=100; higher = stronger)", episodes, fmt=".1f")

    render_two_col([
        ("Reserves by Type", fig_res, _clip(pd.concat([comp, total_res.rename("Total")], axis=1))),
        ("Import Cover", fig_cover, cover),
        ("FX Intervention Gauge", fig_intv, intv),
        ("USD IDR", fig_idr, _clip(idr_full)),
        ("Effective Exchange Rates", fig_eer, eer),
    ])
    st.caption("Reserves: IMF International Liquidity (BI data, gold at national valuation = BI's published basis), "
               "extended with the latest month-end from Bank Indonesia's monetary-indicators page (total only - the "
               "breakdown by type follows on the IMF release). Bank Indonesia's own 'months of "
               "imports' figure also adds government external debt service, so it reads a little lower than this "
               "goods-imports-only measure. The intervention gauge is indicative - reserves also move with valuation "
               "changes and government FX borrowing/repayments.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Current Account (IMF balance of payments, BPM6, USD, quarterly)
# ════════════════════════════════════════════════════════════════════════════════
SERVICES = {"SC": "Transport", "SD": "Travel", "SJ": "Other business services", "SI": "Telecom, computer & IT",
            "SH": "IP charges", "SF": "Insurance & pension", "SG": "Financial services", "SE": "Construction",
            "SA": "Manufacturing services", "SB": "Maintenance & repair", "SK": "Personal & cultural",
            "SL": "Government services"}
CA_COLORS = {"Goods": "#42a5f5", "Services": "#ff9800", "Primary Income": "#ef5350", "Secondary Income": "#26a69a"}

with tabs[3]:
    st.header("Current Account & Balance of Payments")
    with st.spinner("Loading balance of payments…"):
        bop_raw = imf_load(*REQ_BOP, what="balance of payments")

    comps = pd.DataFrame({"Goods": bop(bop_raw, "G", "NETCD_T"), "Services": bop(bop_raw, "S", "NETCD_T"),
                          "Primary Income": bop(bop_raw, "IN1", "NETCD_T"), "Secondary Income": bop(bop_raw, "IN2", "NETCD_T")})
    ca_q = bop(bop_raw, "CAB", "NETCD_T").rename("Current Account")
    comps_c, ca_c = _clip(comps.dropna(how="all")), _clip(ca_q)
    fig_ca = stacked_bars_with_line(comps_c, ca_c, "Current Account Decomposition ($bn, Quarterly)", episodes,
                                    "Current account", colors=CA_COLORS)
    comps4, ca4 = _clip(comps.rolling(4).sum().dropna(how="all")), _clip(ca_q.rolling(4).sum())
    fig_ca4 = stacked_bars_with_line(comps4, ca4, "Current Account — Rolling 4-Quarter Sum ($bn)", episodes,
                                     "Current account (4Q)", colors=CA_COLORS)

    goods = _clip(pd.DataFrame({"Exports (BoP, fob)": bop(bop_raw, "G", "CD_T"), "Imports (BoP, fob)": bop(bop_raw, "G", "DB_T"),
                                "Goods Balance": bop(bop_raw, "G", "NETCD_T")}).dropna(how="all"))
    fig_goods = go.Figure()
    if not goods.empty:
        fig_goods.add_trace(go.Bar(x=goods.index, y=goods["Goods Balance"], name="Goods balance", yaxis="y2",
                                   marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in goods["Goods Balance"]], opacity=0.5))
        fig_goods.add_trace(go.Scatter(x=goods.index, y=goods["Exports (BoP, fob)"], name="Exports", line=dict(color="#42a5f5", width=2)))
        fig_goods.add_trace(go.Scatter(x=goods.index, y=goods["Imports (BoP, fob)"], name="Imports", line=dict(color="#ff9800", width=2)))
    fig_goods.update_layout(**dual_axis_layout("Goods Trade on a BoP Basis ($bn, Quarterly)", "Exports / imports ($bn)", "Balance ($bn)"))
    add_episodes(fig_goods, episodes)

    svc_net = pd.DataFrame({name: bop(bop_raw, code, "NETCD_T") for code, name in SERVICES.items()})
    svc_last4 = svc_net.dropna(how="all").tail(4).sum().sort_values()
    fig_svc_bar = go.Figure(go.Bar(x=svc_last4.values, y=svc_last4.index, orientation="h",
                                   marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in svc_last4.values],
                                   hovertemplate="%{y}: $%{x:.2f}bn<extra></extra>"))
    last4_label = (f"{svc_net.dropna(how='all').index[-4].year}Q{svc_net.dropna(how='all').index[-4].quarter}–"
                   f"{svc_net.dropna(how='all').index[-1].year}Q{svc_net.dropna(how='all').index[-1].quarter}") if len(svc_net.dropna(how="all")) >= 4 else ""
    fig_svc_bar.update_layout(**base_layout(f"Services Balance by Category — Last 4 Quarters ({last4_label}, $bn)", height=480))
    fig_svc_bar.add_vline(x=0, line_dash="dot", line_color="#555")
    fig_svc_bar.update_xaxes(tickprefix="$", ticksuffix="bn")
    fig_svc = line_chart(_clip(svc_net[["Travel", "Transport", "Other business services", "Telecom, computer & IT"]]),
                         "Services Balance — Main Categories ($bn, Quarterly)", episodes, hline=0, fmt=".2f")

    pinc = pd.DataFrame({
        "Direct investment income paid": -bop(bop_raw, "D_F_D4P", "DB_T"),
        "Portfolio income paid": -bop(bop_raw, "P_F_D4P", "DB_T"),
        "Other investment interest paid": -bop(bop_raw, "O_F_D4P", "DB_T"),
        "Compensation paid": -bop(bop_raw, "D1", "DB_T"),
        "Total receipts": bop(bop_raw, "IN1", "CD_T"),
    })
    fig_pinc = stacked_bars_with_line(_clip(pinc.dropna(how="all")), _clip(bop(bop_raw, "IN1", "NETCD_T")),
                                      "Primary Income — Investment Income Paid Abroad vs Received ($bn)", episodes,
                                      "Primary income balance",
                                      colors={"Direct investment income paid": "#ef5350", "Portfolio income paid": "#ff9800",
                                              "Other investment interest paid": "#ab47bc", "Compensation paid": "#8a94a6",
                                              "Total receipts": "#26a69a"})
    sinc = pd.DataFrame({"Workers' remittances received": bop(bop_raw, "D752W_S1W", "CD_T"),
                         "Workers' remittances sent": -bop(bop_raw, "D752W_S1W", "DB_T"),
                         "Other transfers (net)": bop(bop_raw, "IN2", "NETCD_T") - bop(bop_raw, "D752W_S1W", "NETCD_T")})
    fig_sinc = stacked_bars_with_line(_clip(sinc.dropna(how="all")), _clip(bop(bop_raw, "IN2", "NETCD_T")),
                                      "Secondary Income — Migrant-Worker Remittances ($bn)", episodes, "Secondary income balance",
                                      colors={"Workers' remittances received": "#26a69a", "Workers' remittances sent": "#ef5350",
                                              "Other transfers (net)": "#8a94a6"})

    render_two_col([
        ("Current Account Decomposition", fig_ca, _clip(pd.concat([comps, ca_q], axis=1))),
        ("Current Account 4Q Sum", fig_ca4, _clip(pd.concat([comps.rolling(4).sum(), ca_q.rolling(4).sum()], axis=1))),
        ("Goods Trade BoP Basis", fig_goods, goods),
        ("Services by Category", fig_svc_bar, svc_last4.to_frame("Last 4Q $bn")),
        ("Services Main Categories", fig_svc, _clip(svc_net)),
        ("Primary Income", fig_pinc, _clip(pinc)),
        ("Secondary Income", fig_sinc, _clip(sinc)),
    ])
    st.caption("IMF balance of payments (BPM6, compiled by Bank Indonesia). Primary income is Indonesia's structural "
               "drag: dividends and interest paid to foreign owners of Indonesian mines, plantations, factories and "
               "bonds. Services run a deficit mainly on freight & insurance for trade (transport) and business services; "
               "travel swings with tourism and the Hajj/Umrah season.")

    st.markdown('<div class="section-header">Financial Account — How the Current Account Is Funded</div>', unsafe_allow_html=True)
    flows = pd.DataFrame({
        "Direct investment into Indonesia": bop(bop_raw, "D_F", "L_NIL_T"),
        "Foreign buying of debt securities": bop(bop_raw, "P_F3", "L_NIL_T"),
        "Foreign buying of equities": bop(bop_raw, "P_F5", "L_NIL_T"),
        "Other investment liabilities (loans, deposits)": bop(bop_raw, "O_F", "L_NIL_T"),
    })
    fig_flows = stacked_bars_with_line(_clip(flows.dropna(how="all")), _clip(flows.sum(axis=1, min_count=1)),
                                       "Foreign Capital Inflows by Type ($bn, Quarterly)", episodes, "Total foreign inflows",
                                       colors={"Direct investment into Indonesia": "#26a69a", "Foreign buying of debt securities": "#42a5f5",
                                               "Foreign buying of equities": "#ab47bc", "Other investment liabilities (loans, deposits)": "#ff9800"})
    net_fdi = -bop(bop_raw, "D_F", "NNAFANIL_T")   # BPM6 net = assets - liabilities; flip sign -> net inflow
    basic = pd.DataFrame({"Current account (4Q sum)": ca_q.rolling(4).sum(), "Net FDI inflow (4Q sum)": net_fdi.rolling(4).sum()})
    basic["Basic balance (4Q sum)"] = basic.sum(axis=1, min_count=2)
    basic = _clip(basic.dropna(how="all")).round(2)
    fig_basic = line_chart(basic, "Basic Balance — Current Account + Net FDI (Rolling 4Q, $bn)", episodes, hline=0, fmt=".1f",
                           colors={"Current account (4Q sum)": "#ef5350", "Net FDI inflow (4Q sum)": "#26a69a",
                                   "Basic balance (4Q sum)": "#e0e0e0"}, dash={"Current account (4Q sum)": "dot", "Net FDI inflow (4Q sum)": "dot"})
    ovr = pd.DataFrame({"Reserve assets change (+ = accumulation)": bop(bop_raw, "R_F", "A_T"),
                        "Net errors & omissions": bop(bop_raw, "EO", "NETCD_T")})
    fig_ovr = go.Figure()
    o = _clip(ovr.dropna(how="all"))
    for c_, color in zip(o.columns, ["#42a5f5", "#8a94a6"]):
        fig_ovr.add_trace(go.Bar(x=o.index, y=o[c_], name=c_, marker_color=color))
    fig_ovr.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_ovr.update_layout(**base_layout("Overall Balance — Reserve Asset Changes ($bn)"))
    fig_ovr.update_layout(barmode="group")
    add_episodes(fig_ovr, episodes)
    render_two_col([
        ("Foreign Capital Inflows", fig_flows, _clip(flows)),
        ("Basic Balance", fig_basic, basic),
        ("Overall Balance", fig_ovr, o),
    ])
    st.caption("Inflows are foreigners' net purchases of Indonesian assets (BPM6 'net incurrence of liabilities'); "
               "negative = foreigners selling. The basic balance (current account + net FDI) is the stable-funding test: "
               "when it's negative, the rupiah depends on fickle portfolio flows into SBN bonds and equities.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Trade
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Merchandise Trade — Exports & Imports")
    with st.spinner("Loading trade data…"):
        itg_raw = imf_load(*REQ_ITG, what="merchandise trade")
        imts_raw = imf_load(*REQ_IMTS, what="trade by partner")

    ex_m = imf_pick(itg_raw, INDICATOR="XG", TYPE_OF_TRANSFORMATION="FOB_USD") / 1e9
    im_m = imf_pick(itg_raw, INDICATOR="MG", TYPE_OF_TRANSFORMATION="CIF_USD") / 1e9
    tr = pd.DataFrame({"Exports (fob)": ex_m, "Imports (cif)": im_m}).dropna(how="all")
    tr["Trade Balance"] = tr["Exports (fob)"] - tr["Imports (cif)"]
    tr_c = _clip(tr).round(3)
    fig_tr = go.Figure()
    if not tr_c.empty:
        fig_tr.add_trace(go.Bar(x=tr_c.index, y=tr_c["Trade Balance"], name="Trade balance", yaxis="y2", opacity=0.5,
                                marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in tr_c["Trade Balance"].fillna(0)]))
        fig_tr.add_trace(go.Scatter(x=tr_c.index, y=tr_c["Exports (fob)"], name="Exports", line=dict(color="#42a5f5", width=2)))
        fig_tr.add_trace(go.Scatter(x=tr_c.index, y=tr_c["Imports (cif)"], name="Imports", line=dict(color="#ff9800", width=2)))
    fig_tr.update_layout(**dual_axis_layout("Monthly Exports, Imports & Trade Balance ($bn)", "Exports / imports ($bn)", "Balance ($bn)"))
    add_episodes(fig_tr, episodes)
    tr_yoy = _clip(pd.DataFrame({"Exports YoY % (3M avg)": tr["Exports (fob)"].rolling(3).mean().pct_change(12, fill_method=None) * 100,
                                 "Imports YoY % (3M avg)": tr["Imports (cif)"].rolling(3).mean().pct_change(12, fill_method=None) * 100}).dropna(how="all")).round(2)
    fig_tr_yoy = line_chart(tr_yoy, "Export & Import Growth (YoY %, 3-Month Average)", episodes, ysuffix="%", hline=0, fmt=".1f",
                            colors={"Exports YoY % (3M avg)": "#42a5f5", "Imports YoY % (3M avg)": "#ff9800"})
    render_two_col([
        ("Exports Imports Balance", fig_tr, tr_c),
        ("Trade Growth", fig_tr_yoy, tr_yoy),
    ])

    # --- by product (UN Comtrade, HS chapters) --------------------------------
    st.markdown('<div class="section-header">Exports & Imports by Product (HS Chapter, UN Comtrade)</div>', unsafe_allow_html=True)
    ex_hs, im_hs, n_loaded, n_wanted = comtrade_frames()
    if n_loaded < n_wanted:
        st.info(f"Loading product-level trade from UN Comtrade in the background: {n_loaded} of {n_wanted} monthly files so far "
                f"(its free API allows one request at a time, ~10s each). Refresh the page in a minute or two for the rest; "
                f"after that it's cached for 24 hours.")

    def _hs_section(df, flow_name, top_n=8):
        if df.empty:
            return None, None, None
        df = df.sort_index() / 1e9
        top = df.tail(12).sum().sort_values(ascending=False).index[:top_n]
        stacked = df[top].copy()
        stacked["Other"] = df.drop(columns=list(top)).sum(axis=1)
        stacked.columns = [f"{HS_CHAPTERS.get(c, 'HS ' + c)} (HS{c})" if c != "Other" else "Other" for c in stacked.columns]
        fig_area = go.Figure()
        for i, c in enumerate(stacked.columns):
            fig_area.add_trace(go.Bar(x=stacked.index, y=stacked[c], name=c, marker_color=PALETTE[i % len(PALETTE)] if c != "Other" else "#555"))
        fig_area.update_layout(**base_layout(f"{flow_name} by Product — Monthly ($bn, Top {top_n} HS Chapters)", height=520))
        fig_area.update_layout(barmode="stack")
        fig_area.update_yaxes(tickprefix="$", ticksuffix="bn")
        last = df.index[-1]
        ly = last - pd.DateOffset(years=1)
        latest = df.loc[last].sort_values(ascending=False).head(15)
        tbl = pd.DataFrame({"HS": latest.index, "Product": [HS_CHAPTERS.get(c, "HS " + c) for c in latest.index],
                            f"{last:%b %Y} $bn": latest.values.round(3),
                            "Share %": (latest / df.loc[last].sum() * 100).round(1).values,
                            f"YoY % vs {ly:%b %Y}": ((latest / df.loc[ly].reindex(latest.index) - 1) * 100).round(1).values
                            if ly in df.index else [None] * len(latest)})
        fig_bar = go.Figure(go.Bar(x=tbl[f"{last:%b %Y} $bn"][::-1], y=(tbl["Product"] + " (HS" + tbl["HS"] + ")")[::-1], orientation="h",
                                   marker_color="#42a5f5" if flow_name == "Exports" else "#ff9800",
                                   customdata=tbl[f"YoY % vs {ly:%b %Y}"][::-1],
                                   hovertemplate="%{y}<br>$%{x:.2f}bn<br>YoY: %{customdata:+.1f}%<extra></extra>"))
        fig_bar.update_layout(**base_layout(f"{flow_name} — Top 15 Products, {last:%b %Y} ($bn)", height=520))
        fig_bar.update_xaxes(tickprefix="$", ticksuffix="bn")
        return fig_area, fig_bar, tbl

    fx_area, fx_bar, fx_tbl = _hs_section(ex_hs, "Exports")
    fm_area, fm_bar, fm_tbl = _hs_section(im_hs, "Imports")
    charts = []
    if fx_area is not None:
        charts += [("Exports by Product", fx_area, ex_hs / 1e9), ("Top Export Products", fx_bar, fx_tbl)]
    if fm_area is not None:
        charts += [("Imports by Product", fm_area, im_hs / 1e9), ("Top Import Products", fm_bar, fm_tbl)]
    if charts:
        render_two_col(charts)
    st.caption("UN Comtrade monthly data as reported by BPS, by 2-digit HS chapter (~3-month lag). Coal sits in HS27 "
               "alongside oil & gas; palm oil is most of HS15; ferro-nickel/nickel pig iron is in HS72 and nickel matte "
               "& products in HS75 - the downstreaming story. Hover the bars for year-on-year change.")

    # --- by partner (IMF IMTS) ------------------------------------------------
    st.markdown('<div class="section-header">Trade by Partner Country (IMF)</div>', unsafe_allow_html=True)
    ex_p = pd.DataFrame({PARTNERS[c]: imf_pick(imts_raw, INDICATOR="XG_FOB_USD", COUNTERPART_COUNTRY=c) for c in PARTNERS}) / 1e9
    im_p = pd.DataFrame({PARTNERS[c]: imf_pick(imts_raw, INDICATOR="MG_CIF_USD", COUNTERPART_COUNTRY=c) for c in PARTNERS}) / 1e9
    ex_p, im_p = ex_p.dropna(how="all", axis=1), im_p.dropna(how="all", axis=1)

    def _partner_fig(df, title):
        if df.empty:
            return go.Figure(), pd.DataFrame()
        r12 = df.rolling(12, min_periods=12).sum()
        top = r12.iloc[-1].sort_values(ascending=False).index[:8]
        out = _clip(r12[top].dropna(how="all")).round(2)
        return line_chart(out, title, episodes, fmt=".1f"), out

    fig_exp_p, exp_p_df = _partner_fig(ex_p, "Exports by Destination — Rolling 12-Month Sum ($bn)")
    fig_imp_p, imp_p_df = _partner_fig(im_p, "Imports by Origin — Rolling 12-Month Sum ($bn)")
    bal_p = pd.DataFrame()
    if not ex_p.empty and not im_p.empty:
        common = [c for c in ["China", "United States", "India", "Japan", "Singapore"] if c in ex_p and c in im_p]
        bal_p = _clip((ex_p[common] - im_p[common]).rolling(12, min_periods=12).sum().dropna(how="all")).round(2)
    fig_bal_p = line_chart(bal_p, "Bilateral Trade Balance — Rolling 12M ($bn; + = Indonesian surplus)", episodes, hline=0, fmt=".1f")
    share = pd.DataFrame()
    if not ex_p.empty and not ex_m.empty:
        r12_tot = ex_m.rolling(12, min_periods=12).sum()
        share = _clip((ex_p.rolling(12, min_periods=12).sum().div(r12_tot, axis=0) * 100)[[c for c in ["China", "United States", "India", "Japan"] if c in ex_p]].dropna(how="all")).round(2)
    fig_share = line_chart(share, "Share of Total Exports — Rolling 12M (%)", episodes, ysuffix="%", fmt=".1f")
    render_two_col([
        ("Exports by Destination", fig_exp_p, exp_p_df),
        ("Imports by Origin", fig_imp_p, imp_p_df),
        ("Bilateral Trade Balance", fig_bal_p, bal_p),
        ("Export Share by Destination", fig_share, share),
    ])
    china_txt = ""
    if "China" in ex_p and "China" in im_p and not ex_m.empty and not im_m.empty:
        x12 = ex_p["China"].rolling(12, min_periods=12).sum() / ex_m.rolling(12, min_periods=12).sum() * 100
        m12 = im_p["China"].rolling(12, min_periods=12).sum() / im_m.rolling(12, min_periods=12).sum() * 100
        if x12.dropna().size and m12.dropna().size:
            china_txt = (f" Over the latest 12 months China took {x12.dropna().iloc[-1]:.0f}% of Indonesia's exports "
                         f"and supplied {m12.dropna().iloc[-1]:.0f}% of its imports.")
    st.caption("IMF International Merchandise Trade Statistics (BPS-reported, monthly)." + china_txt)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — Growth (IMF quarterly national accounts, BPS source)
# ════════════════════════════════════════════════════════════════════════════════
GDP_COMPONENTS = {"P3_PS": "Private consumption", "P3_S13": "Government consumption", "P51G": "Investment (GFCF)",
                  "P52": "Inventories", "P6": "Exports", "P7": "Imports"}

with tabs[5]:
    st.header("Economic Growth")
    with st.spinner("Loading national accounts…"):
        qn = imf_load(*REQ_QNEA, what="national accounts")

    def qpick(ind, price="Q", sa="NSA", unit="XDC"):
        # QNEA's last dimension (TYPE_OF_TRANSFORMATION) carries the unit code: XDC (rupiah) or IX (index)
        return imf_pick(qn, INDICATOR=ind, PRICE_TYPE=price, S_ADJUSTMENT=sa, TYPE_OF_TRANSFORMATION=unit)

    gdp_r = qpick("B1GQ")
    gdp_r_sa = qpick("B1GQ", sa="SA")
    gdp_n = qpick("B1GQ", price="V")
    defl = qpick("B1GQ", price="PD", sa="SA", unit="IX")
    g = pd.DataFrame({"Real GDP YoY %": gdp_r.pct_change(4, fill_method=None) * 100,
                      "Real GDP QoQ % (SA)": gdp_r_sa.pct_change(fill_method=None) * 100,
                      "Nominal GDP YoY %": gdp_n.pct_change(4, fill_method=None) * 100,
                      "GDP Deflator YoY %": defl.pct_change(4, fill_method=None) * 100})
    g_c = _clip(g.dropna(how="all")).round(2)
    fig_gdp = go.Figure()
    if not g_c.empty:
        q = g_c["Real GDP QoQ % (SA)"].dropna()
        fig_gdp.add_trace(go.Bar(x=q.index, y=q.values, name="QoQ % (SA)", marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in q.values], opacity=0.55))
        y = g_c["Real GDP YoY %"].dropna()
        fig_gdp.add_trace(go.Scatter(x=y.index, y=y.values, name="YoY %", mode="lines+markers", line=dict(color="#42a5f5", width=2.2)))
    fig_gdp.add_hline(y=5, line_dash="dash", line_color="#8a94a6", annotation_text="~5% pre-COVID trend")
    fig_gdp.update_layout(**base_layout("Real GDP Growth"))
    fig_gdp.update_yaxes(ticksuffix="%")
    add_episodes(fig_gdp, episodes)
    fig_nom = line_chart(g_c[["Nominal GDP YoY %", "GDP Deflator YoY %", "Real GDP YoY %"]] if not g_c.empty else g_c,
                         "Nominal vs Real Growth & the GDP Deflator (YoY %)", episodes, ysuffix="%", hline=0, fmt=".1f", mode="lines+markers",
                         dash={"Real GDP YoY %": "dot"})

    comp_lvl = pd.DataFrame({name: qpick(code) for code, name in GDP_COMPONENTS.items()})
    comp_yoy = _clip((comp_lvl.drop(columns=["Inventories"], errors="ignore").pct_change(4, fill_method=None) * 100).dropna(how="all")).round(2)
    fig_comp = line_chart(comp_yoy, "Real Growth by Expenditure Component (YoY %)", episodes, ysuffix="%", hline=0, fmt=".1f", mode="lines+markers")

    contrib = pd.DataFrame(index=gdp_r.index)
    base = gdp_r.shift(4)
    for name in GDP_COMPONENTS.values():
        if name in comp_lvl:
            sign = -1 if name == "Imports" else 1
            contrib[name] = sign * comp_lvl[name].diff(4) / base * 100
    gdp_yoy_series = gdp_r.pct_change(4, fill_method=None) * 100
    contrib["Statistical discrepancy & other"] = gdp_yoy_series - contrib.sum(axis=1, min_count=1)
    contrib_c = _clip(contrib.dropna(how="all")).round(2)
    fig_contrib = stacked_bars_with_line(contrib_c, _clip(gdp_yoy_series), "Contributions to Real GDP Growth (pp, YoY)", episodes,
                                         "Real GDP YoY %", yprefix="", ysuffix="pp",
                                         colors={"Private consumption": "#42a5f5", "Government consumption": "#ab47bc",
                                                 "Investment (GFCF)": "#26a69a", "Inventories": "#8a94a6", "Exports": "#ff9800",
                                                 "Imports": "#ef5350", "Statistical discrepancy & other": "#555"})
    render_two_col([
        ("Real GDP Growth", fig_gdp, g_c),
        ("Contributions to Growth", fig_contrib, contrib_c),
        ("Growth by Component", fig_comp, comp_yoy),
        ("Nominal vs Real Growth", fig_nom, g_c),
    ])
    st.caption("IMF Quarterly National Accounts (BPS data, real = constant prices). Imports enter with a negative sign. "
               "Indonesia has no free monthly activity data (industrial production, PMI and retail sales are "
               "subscription-only or BPS/BI-only), so growth is quarterly.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 — Indonesia Markets
# ════════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("Indonesia Markets")
    st.caption("Equities, rupiah and Indonesia-relevant commodities. Charts follow the global date range; returns and "
               "z-scores use a fixed lookback since 2015.")
    with st.spinner("Loading market data…"):
        idx = fetch_yf_batch(IDX_TICKERS, MARKETS_HIST_START)
        stk = fetch_yf_batch(STOCKS, MARKETS_HIST_START)
        mk = fetch_yf_batch(MACRO_TICKERS, MARKETS_HIST_START)

    def _clip_mkt(df):
        return df[(df.index >= WARMUP_START) & (df.index <= END)] if not df.empty else df

    jci_full, lq_full, eido_full, idr_full = idx["^JKSE"], idx["^JKLQ45"], idx["EIDO"], mk["IDR=X"]
    fig_grid = make_subplots(rows=2, cols=2, subplot_titles=("Jakarta Composite (IDR)", "LQ45 (IDR)", "EIDO — MSCI Indonesia ETF (USD)", "USD/IDR"),
                             vertical_spacing=0.15)
    frames = []
    for (dff, col_name, color, fmt), (row, col) in zip(
            [(jci_full, "Jakarta Composite", "#42a5f5", ",.0f"), (lq_full, "LQ45", "#ab47bc", ",.1f"),
             (eido_full, "EIDO (USD ETF)", "#26a69a", ".2f"), (idr_full, "USD/IDR", "#ff9800", ",.0f")],
            [(1, 1), (1, 2), (2, 1), (2, 2)]):
        if dff.empty:
            continue
        c = _clip_mkt(dff)
        frames.append(c)
        fig_grid.add_trace(go.Scatter(x=c.index, y=c[col_name], name=col_name, line=dict(color=color),
                                      customdata=pd.Series(_z1y(dff[col_name]), index=dff.index).reindex(c.index).values,
                                      hovertemplate=f"%{{x|%Y-%m-%d}}<br>{col_name}: %{{y:{fmt}}}<br>Z (1Y): %{{customdata:.2f}}<extra></extra>"),
                           row=row, col=col)
        add_ma_overlays(fig_grid, dff, col_name, START, END, row=row, col=col)
    fig_grid.update_layout(template=TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, height=560,
                           margin=dict(l=10, r=10, t=50, b=30), showlegend=False)
    fig_grid.update_xaxes(gridcolor=GRID_COLOR)
    fig_grid.update_yaxes(gridcolor=GRID_COLOR)
    add_episodes(fig_grid, episodes, rows=[1, 1, 2, 2], cols=[1, 2, 1, 2])

    usd_view = pd.DataFrame()
    if not jci_full.empty and not idr_full.empty:
        j = pd.concat([jci_full["Jakarta Composite"], idr_full["USD/IDR"]], axis=1).ffill().dropna()
        j = _clip_mkt(j)
        b = j.loc[pd.Timestamp(START):]
        if not b.empty:
            b0 = b.iloc[0]
            usd_view = pd.DataFrame({"JCI in IDR (rebased)": j["Jakarta Composite"] / b0["Jakarta Composite"] * 100,
                                     "JCI in USD (rebased)": (j["Jakarta Composite"] / j["USD/IDR"]) / (b0["Jakarta Composite"] / b0["USD/IDR"]) * 100}).round(2)
    fig_usd = line_chart(usd_view, "Jakarta Composite — Local vs Dollar Returns (rebased = 100)", episodes, hline=100, fmt=".1f",
                         colors={"JCI in IDR (rebased)": "#42a5f5", "JCI in USD (rebased)": "#26a69a"})
    render_two_col([
        ("Indonesia Market Grid", fig_grid, pd.concat(frames, axis=1) if frames else None),
        ("JCI Local vs USD", fig_usd, usd_view),
    ])

    corr_window = st.number_input("Rolling correlation window (trading days)", min_value=10, max_value=252, value=60, step=5, key="idn_corr_window")
    fig_corr = go.Figure()
    corr_df = pd.DataFrame()
    if not jci_full.empty and not idr_full.empty:
        rets = pd.concat([jci_full["Jakarta Composite"], idr_full["USD/IDR"]], axis=1).ffill().dropna().pct_change().dropna()
        c_s = rets["Jakarta Composite"].rolling(int(corr_window)).corr(rets["USD/IDR"])
        c_y = rets["Jakarta Composite"].rolling(252).corr(rets["USD/IDR"])
        corr_df = _clip_mkt(pd.DataFrame({f"{int(corr_window)}D": c_s, "1Y": c_y}).dropna()).round(3)
        for col_name, color in [(f"{int(corr_window)}D", "#42a5f5"), ("1Y", "#ab47bc")]:
            fig_corr.add_trace(go.Scatter(x=corr_df.index, y=corr_df[col_name], name=col_name, line=dict(color=color)))
    fig_corr.add_hline(y=0, line_dash="dot", line_color="#555")
    fig_corr.update_layout(**base_layout("JCI vs USD/IDR — Rolling Correlation of Daily Returns"))
    add_episodes(fig_corr, episodes)

    fig_cm = make_subplots(rows=2, cols=2, subplot_titles=("Brent ($/bbl)", "Iron Ore ($/t)", "Aluminium ($/t)", "Gold ($/oz)"), vertical_spacing=0.15)
    cm_frames = []
    for (t, col_name, color), (row, col) in zip([("BZ=F", "Brent", "#ff9800"), ("TIO=F", "Iron Ore", "#ef5350"),
                                                 ("ALI=F", "Aluminium", "#90a4d4"), ("GC=F", "Gold", "#eda100")],
                                                [(1, 1), (1, 2), (2, 1), (2, 2)]):
        dff = mk[t]
        if dff.empty:
            continue
        c = _clip_mkt(dff)
        cm_frames.append(c)
        fig_cm.add_trace(go.Scatter(x=c.index, y=c[col_name], name=col_name, line=dict(color=color)), row=row, col=col)
        add_ma_overlays(fig_cm, dff, col_name, START, END, row=row, col=col)
    fig_cm.update_layout(template=TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, height=560,
                         margin=dict(l=10, r=10, t=50, b=30), showlegend=False)
    fig_cm.update_xaxes(gridcolor=GRID_COLOR)
    fig_cm.update_yaxes(gridcolor=GRID_COLOR)
    add_episodes(fig_cm, episodes, rows=[1, 1, 2, 2], cols=[1, 2, 1, 2])
    render_two_col([
        ("JCI vs Rupiah Correlation", fig_corr, corr_df),
        ("Commodities", fig_cm, pd.concat(cm_frames, axis=1) if cm_frames else None),
    ])
    st.caption("Negative JCI-vs-USD/IDR correlation = stocks fall when the rupiah weakens (foreign outflows hit both). "
               "Yahoo has no coal, palm oil or nickel futures - those show up as export values in the Trade tab.")

    # ── Foreign portfolio flows: bonds vs equities (BoP) ────────────────────────
    st.markdown('<div class="section-header">Foreign Portfolio Flows — Bonds vs Equities</div>', unsafe_allow_html=True)
    with st.spinner("Loading portfolio flows…"):
        bop_raw_m = imf_load(*REQ_BOP, what="balance of payments")
    pf = pd.DataFrame({
        "Bonds — foreign net buying (+) / selling (−)": bop(bop_raw_m, "P_F3", "L_NIL_T"),
        "Equities — foreign net buying (+) / selling (−)": bop(bop_raw_m, "P_F5", "L_NIL_T"),
        "Residents' portfolio investment abroad (outflow)": -bop(bop_raw_m, "P_F", "A_NFA_T"),
    })
    net_pf = -bop(bop_raw_m, "P_F", "NNAFANIL_T")   # BPM6 net = assets − liabilities; flip -> net inflow
    pf_c = _clip(pf.dropna(how="all"))
    fig_pf = stacked_bars_with_line(pf_c, _clip(net_pf), "Portfolio Flows — Bonds vs Equities ($bn, Quarterly)", episodes,
                                    "Net portfolio inflow",
                                    colors={"Bonds — foreign net buying (+) / selling (−)": "#42a5f5",
                                            "Equities — foreign net buying (+) / selling (−)": "#ab47bc",
                                            "Residents' portfolio investment abroad (outflow)": "#8a94a6"})
    pf_cum = _clip(pd.DataFrame({"Bonds (rolling 4Q sum)": pf.iloc[:, 0].rolling(4).sum(),
                                 "Equities (rolling 4Q sum)": pf.iloc[:, 1].rolling(4).sum(),
                                 "Net portfolio inflow (rolling 4Q sum)": net_pf.rolling(4).sum()}).dropna(how="all")).round(2)
    fig_pf_cum = line_chart(pf_cum, "Foreign Bond vs Equity Flows — Rolling 4-Quarter Sum ($bn)", episodes, hline=0, fmt=".1f",
                            colors={"Bonds (rolling 4Q sum)": "#42a5f5", "Equities (rolling 4Q sum)": "#ab47bc",
                                    "Net portfolio inflow (rolling 4Q sum)": "#e0e0e0"},
                            dash={"Net portfolio inflow (rolling 4Q sum)": "dot"})
    render_two_col([
        ("Portfolio Flows Bonds vs Equities", fig_pf, _clip(pd.concat([pf, net_pf.rename("Net portfolio inflow")], axis=1))),
        ("Portfolio Flows 4Q Sum", fig_pf_cum, pf_cum),
    ])
    st.caption("IMF balance of payments (Bank Indonesia), quarterly: 'bonds' = non-residents' net purchases of Indonesian "
               "debt securities (SBN, SRBI and corporate/global bonds); 'equities' = net purchases of Indonesian shares. "
               "Free sources don't publish the weekly/daily foreign flow data (BI's weekly releases and IDX/DJPPR "
               "files aren't available as clean data feeds), so this is the highest-frequency free view.")

    # ── Cross-asset correlation matrix ───────────────────────────────────────────
    st.markdown('<div class="section-header">Cross-Asset Correlation Matrix</div>', unsafe_allow_html=True)
    vol = fetch_yf_batch(VOL_TICKERS, MARKETS_HIST_START)
    sbn_m = load_sbn_history(record=False)
    cm_c1, cm_c2 = st.columns([2, 1])
    with cm_c1:
        cm_win = int(st.number_input("Correlation matrix window (observations, ending at the date range's end)", min_value=10,
                                     max_value=2500, value=63, step=5, key="idn_corr_matrix_window",
                                     help="Daily: 21 ≈ 1M, 63 ≈ 3M, 126 ≈ 6M, 252 ≈ 1Y. Weekly: 26 ≈ 6M, 52 ≈ 1Y. "
                                          "SBN yields join once 20 days are recorded."))
    with cm_c2:
        cm_freq = st.radio("Change horizon", ["Weekly", "Daily"], index=0, horizontal=True, key="idn_corr_matrix_freq",
                           help="Jakarta-close series (USD/IDR, SBN) and New-York-close series (VIX, MOVE, gold, oil) "
                                "record the same shock on different calendar days, so daily correlations across them "
                                "are understated. Weekly (Friday-to-Friday) changes remove most of that mismatch.")
    levels = {}
    for dff, col_name, label in [(mk["IDR=X"], "USD/IDR", "USD/IDR"), (mk["GC=F"], "Gold", "Gold"), (mk["BZ=F"], "Brent", "Oil (Brent)"),
                                 (vol["^VIX"], "VIX", "VIX"), (vol["^MOVE"], "MOVE", "MOVE")]:
        if not dff.empty:
            levels[label] = dff[col_name]
    lv = pd.DataFrame(levels).sort_index().ffill()
    lv = lv[lv.index <= pd.Timestamp(END)]
    sbn_ok = len(sbn_m.dropna(how="all")) - 1 >= MIN_BETA_OBS if not sbn_m.empty else False
    if sbn_ok:
        # Sample = recorded SBN dates; every series' change is measured between the same consecutive dates
        y = sbn_m[["1Y", "5Y", "10Y"]].rename(columns=lambda c: f"IDR {c}")
        y = y[y.index <= pd.Timestamp(END)]
        lv = pd.concat([y, lv.reindex(lv.index.union(y.index)).ffill().reindex(y.index)], axis=1)
    else:
        lv = lv.reindex(mk["IDR=X"].index if not mk["IDR=X"].empty else lv.index).ffill()
    if cm_freq == "Weekly":
        lv = lv.resample("W-FRI").last()
    chg = pd.DataFrame(index=lv.index)
    for c_ in lv.columns:
        if c_.startswith("IDR "):
            chg[c_] = lv[c_].diff() * 100                                 # bps
        elif c_ in ("VIX", "MOVE"):
            chg[c_] = lv[c_].diff()                                       # vol points
        else:
            chg[c_] = lv[c_].pct_change(fill_method=None) * 100           # % return
    window = chg.dropna(how="all").tail(cm_win)
    corr = window.corr(min_periods=max(8, cm_win // 3)).round(2)
    fig_corrm = go.Figure()
    if not corr.empty:
        nobs = window.notna().astype(int).T.dot(window.notna().astype(int))
        names = list(corr.columns)
        fig_corrm.add_trace(go.Heatmap(z=corr.values, x=names, y=names, zmin=-1, zmax=1,
                                       colorscale=[[0, "#ef5350"], [0.5, "#161b26"], [1, "#26a69a"]],
                                       text=corr.map(lambda v: "" if pd.isna(v) else f"{v:+.2f}").values, texttemplate="%{text}",
                                       customdata=nobs.reindex(index=names, columns=names).values,
                                       hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<br>Obs: %{customdata}<extra></extra>",
                                       colorbar=dict(title="ρ", tickvals=[-1, -0.5, 0, 0.5, 1])))
        ws, we = window.index.min(), window.index.max()
        fig_corrm.update_layout(**base_layout(f"Correlation of {cm_freq} Changes — {len(window)} Obs ({ws:%Y-%m-%d} to {we:%Y-%m-%d})", height=560))
        fig_corrm.update_yaxes(autorange="reversed", showgrid=False)
        fig_corrm.update_xaxes(showgrid=False)
    st.plotly_chart(fig_corrm, use_container_width=True, key="chart_idn_corr_matrix")
    csv_download(corr, f"idn_cross_asset_correlation_{cm_win}d")
    if not sbn_ok:
        n_sbn = max(len(sbn_m) - 1, 0) if not sbn_m.empty else 0
        st.info(f"IDR 1Y / 5Y / 10Y government yields will join this matrix once {MIN_BETA_OBS} daily changes are recorded "
                f"from PHEI ({n_sbn} so far). Until then the matrix covers USD/IDR, Gold, Oil, VIX and MOVE.")
    st.caption("Changes: bps for SBN yields, % returns for USD/IDR (+ = weaker rupiah), gold and Brent, point "
               "changes for VIX and MOVE (ICE BofA MOVE = Treasury implied volatility, the bond market's VIX). Note sign: "
               "IDR yield vs USD/IDR > 0 means yields rise as the rupiah weakens - the classic EM stress pattern.")

    st.markdown('<div class="section-header">Index & Large-Cap Returns and Z-Scores</div>', unsafe_allow_html=True)

    def _returns_and_z(full_map, names):
        rows, zrows = [], []
        for t, name in names.items():
            full = full_map.get(t, pd.DataFrame())
            if full.empty:
                continue
            s = full[name].dropna()
            rows.append({"Name": name, "Level": s.iloc[-1], **{f"{k} Return %": v for k, v in calculate_period_returns(s).items()}})
            zrows.append({"Name": name, **{f"{k} Z-Score": v for k, v in calculate_price_zscores(s).items()}})
        ret = pd.DataFrame(rows)
        if ret.empty:
            return ret, pd.DataFrame()
        ret = ret.sort_values("1M Return %").round(3)
        z = pd.DataFrame(zrows).set_index("Name").reindex(ret["Name"]).reset_index().round(3)
        return ret, z

    all_map = {**idx, **stk}
    all_names = {**IDX_TICKERS, **STOCKS}
    ret, z = _returns_and_z(all_map, all_names)
    f_ret, f_z = go.Figure(), go.Figure()
    for label, color in [("1M", "#42a5f5"), ("3M", "#ab47bc"), ("1Y", "#ff9800")]:
        if not ret.empty:
            f_ret.add_trace(go.Bar(x=ret[f"{label} Return %"], y=ret["Name"], name=f"{label} Return %", orientation="h", marker_color=color))
            f_z.add_trace(go.Bar(x=z[f"{label} Z-Score"], y=z["Name"], name=f"{label} Z-Score", orientation="h", marker_color=color))
    for f, t in [(f_ret, "Returns — 1M / 3M / 1Y (%, local currency)"), (f_z, "Price Z-Scores (vs Own Trailing History)")]:
        f.add_vline(x=0, line_dash="dot", line_color="#555")
        f.update_layout(**base_layout(t, height=620))
        f.update_layout(barmode="group")
    f_ret.update_xaxes(ticksuffix="%")
    render_two_col([("Index and Stock Returns", f_ret, ret), ("Index and Stock Z-Scores", f_z, z)])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 8 — Economic Calendar
# ════════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("Economic Calendar")
    st.caption("Live Investing.com economic calendar widget — Indonesia (IDR) events, all importance levels, times in "
               "GMT+8 (Singapore). Use the widget's own controls to change dates or timezone.")
    INVESTING_CALENDAR_SRC = ("https://sslecal2.investing.com?"
        "columns=exc_flags,exc_currency,exc_importance,exc_actual,exc_forecast,exc_previous"
        "&importance=1,2,3&features=datepicker,timezone&countries=48&calType=week&timeZone=113&lang=1")
    components.html(f"""
        <iframe src="{INVESTING_CALENDAR_SRC}" width="100%" height="700" frameborder="0"
                allowtransparency="true" marginwidth="0" marginheight="0"></iframe>
        <div style="font-family: Arial, Helvetica, sans-serif; text-align:right; margin-top:4px;">
            <span style="font-size: 11px; color: #888;">Real Time Economic Calendar provided by
            <a href="https://www.investing.com/" rel="nofollow" target="_blank"
               style="color:#06529D; font-weight:bold;">Investing.com</a>.</span>
        </div>
    """, height=740, scrolling=True)

st.markdown("---")
st.caption("Data: IMF Data API · BIS · UN Comtrade · Yahoo Finance · Shaded areas = stress episodes (2013 taper tantrum, "
           "2018 EM sell-off, 2020 COVID, 2026 Strait of Hormuz closure) · Refresh: 6hr cache (1hr markets, 24hr Comtrade)")
