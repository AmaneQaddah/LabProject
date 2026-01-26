# recommender.py
import os
import glob
import math
from datetime import timedelta
from typing import Tuple, Dict, Any, Optional

import pandas as pd
from functools import lru_cache

# ----------------------------
# Configuration (data paths)
# ----------------------------
AIRBNB_PATH = os.getenv("AIRBNB_PARQUET_PATH", "data/airbnb_sample_parquet")
HOLIDAYS_PATH = os.getenv("HOLIDAYS_PARQUET_PATH", "data/holidays_mapped_parquet")
WORLDCUP_PATH = os.getenv("WORLDCUP_PARQUET_PATH", "data/worldcup_mapped_parquet")  # optional

# ----------------------------
# Helpers: read Spark parquet folder (ALL part files)
# ----------------------------
def _read_spark_parquet_folder(folder_path: str) -> pd.DataFrame:
    part_files = sorted(
        glob.glob(os.path.join(folder_path, "part-*.parquet")) +
        glob.glob(os.path.join(folder_path, "part-*"))
    )
    part_files = [p for p in part_files if os.path.basename(p).startswith("part-")]
    if not part_files:
        raise FileNotFoundError(f"No parquet part files found in: {folder_path}")

    dfs = [pd.read_parquet(p) for p in part_files]
    return pd.concat(dfs, ignore_index=True)


    dfs = []
    for p in part_files:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception:
            pass

    if not dfs:
        if required:
            raise FileNotFoundError(f"Found part files but could not read parquet in: {folder_path}")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)

# ----------------------------
# Major keywords (token-style)
# ----------------------------
MAJOR_KEYWORDS = [
    "newyear", "christmas", "easter", "thanksgiving",
    "ramadan", "eid", "hajj", "arafat", "ashura", "muharram", "mawlid", "hijra", "qadr", "miraj",
    "passover", "rosh", "kippur", "sukkot", "shavuot", "hanukkah", "purim",
    "diwali", "holi", "navratri", "dussehra", "shivaratri", "ramnavami", "ganesh", "vaisakhi", "gurunanak",
    "vesak", "wesak", "visakha", "buddha",
    "lunar", "springfestival", "seollal", "tet", "chuseok",
    "carnival", "songkran", "midautumn", "dragonboat", "midsummer",
    "independence", "national", "republic", "constitution", "liberation", "victory",
    "revolution", "unification", "labor", "mayday",
]

def _norm_tokens(s: str) -> str:
    s = (s or "").lower()
    return "".join([ch for ch in s if ch.isalnum()])

def _is_major_event(event_name: str, is_worldcup: bool) -> bool:
    if is_worldcup:
        return True
    n = _norm_tokens(event_name)
    return any(k in n for k in MAJOR_KEYWORDS)

def _weekend_boost(d: pd.Timestamp) -> float:
    wd = d.weekday()  # Mon=0..Sun=6
    return 1.18 if wd in (4, 5, 6) else 1.0

def _season_boost(d: pd.Timestamp) -> float:
    m = d.month
    if m in (6, 7, 8):
        return 1.08
    if m in (11, 12):
        return 1.06
    return 1.0

def _type_weight(holiday_type: Optional[str], is_worldcup: bool) -> float:
    if is_worldcup:
        return 1.30

    ht = (holiday_type or "")
    ht = str(ht).lower().strip()

    if "public holiday" in ht: return 1.00
    if "national holiday" in ht: return 1.00
    if "statutory holiday" in ht: return 1.00
    if "official holiday" in ht: return 0.95
    if "federal holiday" in ht: return 0.95

    if "bank holiday" in ht: return 0.90
    if "government holiday" in ht: return 0.90
    if "regional government" in ht: return 0.85
    if "state holiday" in ht: return 0.85

    if "half-day" in ht: return 0.75
    if "special" in ht: return 0.75
    if "silent day" in ht: return 0.70

    return 0.75

def _lead_days_used(base_lead: int, is_major: bool, is_worldcup: bool) -> int:
    if is_worldcup:
        return max(base_lead, 28)
    if is_major:
        return max(base_lead, 21)
    return base_lead

def _why_tags(row: pd.Series) -> str:
    tags = []
    if row["is_worldcup"]:
        tags.append("World Cup event")
    elif row["is_major"]:
        tags.append("Major holiday")
    if row["is_weekend"]:
        tags.append("Weekend")
    tags.append(f"In {int(row['days_to_event'])} days")
    tags.append(f"Score={row['score']:.3f}")
    if pd.notna(row.get("holiday_type")) and str(row.get("holiday_type")).strip():
        tags.append(f"Type: {row['holiday_type']}")
    return ", ".join(tags)

# ----------------------------
# Pricing uplift heuristic (keeps urgency by days_to_event)
# ----------------------------
def _price_uplift_pct(score: float, days_to_event: int, is_worldcup: bool, is_major: bool, is_weekend: bool, holiday_type: Optional[str]) -> float:
    score_norm = 1.0 - math.exp(-max(score, 0.0) / 1.2)  # 0..1
    pct = score_norm * 22.0

    if is_major:
        pct += 6.0
    if is_worldcup:
        pct += 10.0
    if is_weekend:
        pct += 4.0

    # urgency (only affects pricing, not ranking score)
    if days_to_event <= 14:
        pct += 6.0
    elif days_to_event <= 30:
        pct += 3.0

    ht = (holiday_type or "").lower()
    if "public holiday" in ht or "national holiday" in ht or "statutory holiday" in ht:
        pct += 3.0
    elif "bank holiday" in ht:
        pct += 1.0
    elif "half-day" in ht or "silent day" in ht:
        pct -= 2.0

    pct = max(0.0, min(pct, 45.0))
    return round(pct, 1)

def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip().replace("$", "").replace(",", "")
        x = float(v)
        if math.isnan(x) or x <= 0:
            return None
        return x
    except Exception:
        return None

def _extract_base_price(airbnb_row: pd.Series) -> Optional[float]:
    for col in ["nightly_price", "price", "base_price", "avg_price"]:
        if col in airbnb_row.index:
            x = _safe_float(airbnb_row.get(col))
            if x is not None:
                return x
    return None

@lru_cache(maxsize=1)
def _load_events_cached() -> tuple[pd.DataFrame, pd.DataFrame]:
    airbnb = _read_spark_parquet_folder(AIRBNB_PATH, required=True)
    holidays = _read_spark_parquet_folder(HOLIDAYS_PATH, required=True)
    worldcup = _read_spark_parquet_folder(WORLDCUP_PATH, required=False)

    # Airbnb normalize
    if "property_id" not in airbnb.columns or "country_mapped" not in airbnb.columns:
        raise ValueError("Airbnb parquet must include columns: property_id, country_mapped")
    airbnb["property_id"] = airbnb["property_id"].astype(str).str.strip()
    airbnb["country_mapped"] = airbnb["country_mapped"].astype(str).str.strip().str.lower()

    # Holidays normalize
    if "country_mapped" not in holidays.columns:
        raise ValueError("Holidays parquet must include column: country_mapped")
    holidays["country_mapped"] = holidays["country_mapped"].astype(str).str.strip().str.lower()

    h_date_col = "holiday_date" if "holiday_date" in holidays.columns else ("date" if "date" in holidays.columns else None)
    if h_date_col is None:
        raise ValueError("Holidays dataset must have 'holiday_date' or 'date' column.")

    if "holiday_name" not in holidays.columns and "name" in holidays.columns:
        holidays["holiday_name"] = holidays["name"].astype(str)
    if "holiday_name" not in holidays.columns:
        holidays["holiday_name"] = "Holiday"

    holidays["date"] = pd.to_datetime(holidays[h_date_col], errors="coerce")
    holidays = holidays[holidays["date"].notna()].copy()
    holidays["event_name"] = holidays["holiday_name"].astype(str)
    holidays["event_type"] = "holiday"
    if "holiday_type" not in holidays.columns:
        holidays["holiday_type"] = None

    h_events = holidays[["country_mapped", "event_name", "date", "event_type", "holiday_type"]].copy()

    # World Cup normalize (optional)
    wc_events = pd.DataFrame(columns=["country_mapped", "event_name", "date", "event_type", "holiday_type"])
    if not worldcup.empty:
        if "country_mapped" not in worldcup.columns and "country" in worldcup.columns:
            worldcup = worldcup.rename(columns={"country": "country_mapped"})
        if "country_mapped" in worldcup.columns:
            worldcup["country_mapped"] = worldcup["country_mapped"].astype(str).str.strip().str.lower()
            wc_date_col = "date" if "date" in worldcup.columns else ("date_dt" if "date_dt" in worldcup.columns else None)
            if wc_date_col is not None:
                worldcup["date"] = pd.to_datetime(worldcup[wc_date_col], errors="coerce")
                worldcup = worldcup[worldcup["date"].notna()].copy()
                if "event_name" not in worldcup.columns:
                    worldcup["event_name"] = "World Cup"
                worldcup["event_type"] = "worldcup"
                worldcup["holiday_type"] = None
                wc_events = worldcup[["country_mapped", "event_name", "date", "event_type", "holiday_type"]].copy()

    events = pd.concat([h_events, wc_events], ignore_index=True)
    return airbnb, events

def recommend_publish_date(
    property_id: str,
    today,
    base_lead_days: int = 14,
    lookahead_days: int = 365,
    top_k_by_time: int = 60,
    alternatives_n: int = 3,
    base_price: Optional[float] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:

    try:
        airbnb, events = _load_events_cached()
    except Exception as e:
        return None, f"Dataset load error: {e}"

    pid = str(property_id).strip()
    row = airbnb[airbnb["property_id"] == pid]
    if row.empty:
        return None, "Property not found"

    airbnb_row = row.iloc[0]
    country = str(airbnb_row["country_mapped"]).strip().lower()
    if not country:
        return None, "Country mapping missing for this property"

    # base price (user > dataset > None)
    user_price = _safe_float(base_price)
    data_price = _extract_base_price(airbnb_row)
    base_price_used = user_price if user_price is not None else data_price

    today_dt = pd.to_datetime(today).normalize()
    max_dt = today_dt + pd.Timedelta(days=int(lookahead_days))

    e = events[events["country_mapped"] == country].copy()
    if e.empty:
        return None, "No events (holidays/worldcup) for this country"

    # window filter (still needed)
    e = e[(e["date"] >= today_dt) & (e["date"] <= max_dt)].copy()
    if e.empty:
        return None, "No upcoming events in the selected lookahead window"

    # take next K by time for efficiency
    e = e.sort_values("date").head(int(top_k_by_time)).copy()

    # features
    e["days_to_event"] = (e["date"] - today_dt).dt.days
    e["is_worldcup"] = e["event_type"].astype(str).str.lower().eq("worldcup")
    e["is_major"] = e.apply(lambda r: _is_major_event(str(r["event_name"]), bool(r["is_worldcup"])), axis=1)

    e["major_boost"] = 1.0
    e.loc[e["is_major"], "major_boost"] = 1.45
    e.loc[e["is_worldcup"], "major_boost"] = 1.90

    e["weekend_boost"] = e["date"].map(_weekend_boost)
    e["season_boost"] = e["date"].map(_season_boost)
    e["type_weight"] = e.apply(lambda r: _type_weight(r.get("holiday_type"), bool(r["is_worldcup"])), axis=1)

    # ----------------------------
    # SCORE WITHOUT TIME EFFECT
    # ----------------------------
    e["score"] = e["major_boost"] * e["type_weight"] * e["weekend_boost"] * e["season_boost"]

    # lead days + publish date
    e["is_weekend"] = e["date"].dt.weekday.isin([4, 5, 6])
    e["lead_days_used"] = e.apply(
        lambda r: _lead_days_used(int(base_lead_days), bool(r["is_major"]), bool(r["is_worldcup"])),
        axis=1
    )

    e["publish_date_raw"] = e["date"] - pd.to_timedelta(e["lead_days_used"], unit="D")
    e["publish_date"] = e["publish_date_raw"]
    e.loc[e["publish_date"] < today_dt, "publish_date"] = today_dt

    e["campaign_start"] = e["publish_date"]
    e["campaign_peak"] = e["date"]
    e["campaign_end"] = e["date"] + pd.Timedelta(days=2)

    e["why"] = e.apply(_why_tags, axis=1)

    # pricing uplift
    e["price_uplift_pct"] = e.apply(
        lambda r: _price_uplift_pct(
            score=float(r["score"]),
            days_to_event=int(r["days_to_event"]),
            is_worldcup=bool(r["is_worldcup"]),
            is_major=bool(r["is_major"]),
            is_weekend=bool(r["is_weekend"]),
            holiday_type=r.get("holiday_type"),
        ),
        axis=1
    )
    if base_price_used is not None:
        e["recommended_price"] = (base_price_used * (1.0 + e["price_uplift_pct"] / 100.0)).round(2)
    else:
        e["recommended_price"] = None

    # ranking: score desc, then date asc
    ranked = e.sort_values(["score", "date"], ascending=[False, True]).reset_index(drop=True)
    best = ranked.iloc[0]

    def _pack(r: pd.Series) -> Dict[str, Any]:
        return {
            "event_type": r["event_type"],
            "event": r["event_name"],
            "event_date": r["date"].date(),
            "publish_date": r["publish_date"].date(),
            "score": float(r["score"]),
            "why": r["why"],
            "campaign_start": r["campaign_start"].date(),
            "campaign_peak": r["campaign_peak"].date(),
            "campaign_end": r["campaign_end"].date(),
            "price_uplift_pct": float(r["price_uplift_pct"]),
            "base_price_used": float(base_price_used) if base_price_used is not None else None,
            "recommended_price": float(r["recommended_price"]) if base_price_used is not None else None,
            # lead_days_used is kept internally but not required to show in UI
            "lead_days_used": int(r["lead_days_used"]),
        }

    alternatives = []
    for i in range(1, min(1 + int(alternatives_n), len(ranked))):
        alternatives.append(_pack(ranked.iloc[i]))

    return {
        "property_id": pid,
        "country": country,
        "best": _pack(best),
        "alternatives": alternatives,
    }, None

