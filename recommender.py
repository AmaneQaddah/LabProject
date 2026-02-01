# recommender.py
import os
import math
from typing import Tuple, Dict, Any, Optional, List
from functools import lru_cache

import pandas as pd

# =========================
# Data paths (CSV for deployment)
# =========================
AIRBNB_PATH = os.getenv("AIRBNB_CSV_PATH", "data/airbnb_sample_100.csv")
EVENTS_PATH = os.getenv("EVENTS_CSV_PATH", "data/events_sample_100.csv")

# =========================
# Major keywords (token-style) - same spirit as Databricks
# =========================
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
    return "".join(ch for ch in s if ch.isalnum())

def _is_major_event(event_name: str) -> bool:
    n = _norm_tokens(event_name)
    return any(k in n for k in MAJOR_KEYWORDS)

def _weekend_boost(d: pd.Timestamp) -> float:
    wd = d.weekday()  # Mon=0..Sun=6
    return 1.18 if wd in (4, 5, 6) else 1.0  # Fri/Sat/Sun

def _season_boost(d: pd.Timestamp) -> float:
    m = d.month
    if m in (6, 7, 8):
        return 1.08
    if m in (11, 12):
        return 1.06
    return 1.0

def _type_weight(holiday_type: Optional[str]) -> float:
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

def _lead_days_used(base_lead: int, event_type: str, is_major: bool) -> int:
    et = (event_type or "").lower().strip()
    if et == "worldcup":
        return max(base_lead, 28)
    if is_major:
        return max(base_lead, 21)
    return base_lead

def _why_tags(row: pd.Series) -> str:
    tags: List[str] = []
    if row.get("event_type", "") == "worldcup":
        tags.append("World Cup event")
    elif row["is_major"]:
        tags.append("Major holiday")

    if row["is_weekend"]:
        tags.append("Weekend")

    tags.append(f"In {int(row['days_to_event'])} days")
    tags.append(f"Score={row['score']:.3f}")

    if pd.notna(row.get("holiday_type")) and str(row.get("holiday_type")).strip():
        tags.append(f"Type: {row['holiday_type']}")

    if row.get("rating_value") is not None and not pd.isna(row.get("rating_value")):
        tags.append(f"Rating={float(row['rating_value']):.2f}")

    return ", ".join(tags)

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

def _extract_first_number(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    import re
    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _extract_rating(airbnb_row: pd.Series) -> Optional[float]:
    # Prefer already-clean rating_value if exists, else try common candidates
    candidates = [
        "rating_value", "rating", "ratings", "review_rating", "avg_rating", "overall_rating",
        "category_rating"
    ]
    for c in candidates:
        if c in airbnb_row.index:
            v = _extract_first_number(airbnb_row.get(c))
            if v is None:
                continue
            # clamp to plausible rating range
            if 0 < v <= 5.0:
                return float(v)
            # sometimes ratings are given as 90/100 etc -> ignore
    return None

def _quality_factor_from_rating(r: Optional[float]) -> float:
    # Simple, bounded, interpretable: 0.95..1.15
    if r is None:
        return 1.00
    if r >= 4.8:
        return 1.15
    if r >= 4.5:
        return 1.10
    if r >= 4.0:
        return 1.05
    if r >= 3.5:
        return 1.00
    return 0.95

def _caps_by_event_type(event_type: str, is_major: bool) -> float:
    et = (event_type or "").lower().strip()
    if et == "worldcup":
        return 40.0
    if is_major:
        return 25.0
    return 15.0

def _uplift_from_score(score: float, cap: float, quality_factor: float) -> float:
    # Monotonic mapping score -> [0..cap], then multiply by quality factor and clamp to cap
    score = max(float(score), 0.0)

    # convert to 0..1 smoothly (works well without calibration)
    score_norm = 1.0 - math.exp(-score / 1.2)  # 0..1

    pct = score_norm * cap
    pct = pct * float(quality_factor)

    # bound to [0..cap]
    pct = max(0.0, min(pct, cap))
    return round(pct, 1)

def _extract_base_price(airbnb_row: pd.Series) -> Optional[float]:
    for col in ["price_clean", "total_price_clean", "nightly_price", "price", "base_price", "avg_price"]:
        if col in airbnb_row.index:
            x = _safe_float(airbnb_row.get(col))
            if x is not None:
                return x
    return None

@lru_cache(maxsize=1)
def _load_airbnb() -> pd.DataFrame:
    if not os.path.exists(AIRBNB_PATH):
        raise FileNotFoundError(f"Missing file: {AIRBNB_PATH}")

    df = pd.read_csv(AIRBNB_PATH)

    if "property_id" not in df.columns or "country_mapped" not in df.columns:
        raise ValueError("Airbnb CSV must include columns: property_id, country_mapped")

    df["property_id"] = df["property_id"].astype(str).str.strip()
    df["country_mapped"] = df["country_mapped"].astype(str).str.strip().str.lower()
    return df

@lru_cache(maxsize=1)
def _load_events() -> pd.DataFrame:
    if not os.path.exists(EVENTS_PATH):
        raise FileNotFoundError(f"Missing file: {EVENTS_PATH}")

    df = pd.read_csv(EVENTS_PATH)

    if "country_mapped" not in df.columns:
        raise ValueError("Events CSV must include column: country_mapped")

    df["country_mapped"] = df["country_mapped"].astype(str).str.strip().str.lower()

    # date column
    date_col = None
    for c in ["date", "holiday_date", "event_date"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError("Events CSV must have a date column: date / holiday_date / event_date")

    # event name column
    if "event_name" not in df.columns:
        if "holiday_name" in df.columns:
            df["event_name"] = df["holiday_name"].astype(str)
        elif "name" in df.columns:
            df["event_name"] = df["name"].astype(str)
        else:
            df["event_name"] = "Event"

    # event type column
    if "event_type" not in df.columns:
        df["event_type"] = "holiday"
    else:
        df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()

    # holiday_type optional
    if "holiday_type" not in df.columns:
        df["holiday_type"] = None

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df["date"].notna()].copy()

    return df[["country_mapped", "event_name", "date", "event_type", "holiday_type"]].copy()

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
        airbnb = _load_airbnb()
        events = _load_events()
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

    # rating + quality factor (quality)
    rating_value = _extract_rating(airbnb_row)
    quality_factor = _quality_factor_from_rating(rating_value)

    today_dt = pd.to_datetime(today).normalize()
    max_dt = today_dt + pd.Timedelta(days=int(lookahead_days))

    e = events[events["country_mapped"] == country].copy()
    if e.empty:
        return None, "No events for this country"

    e = e[(e["date"] >= today_dt) & (e["date"] <= max_dt)].copy()
    if e.empty:
        return None, "No upcoming events in the selected lookahead window"

    # Limit to nearest N events for speed/clarity
    e = e.sort_values("date").head(int(top_k_by_time)).copy()

    e["days_to_event"] = (e["date"] - today_dt).dt.days
    e["is_weekend"] = e["date"].dt.weekday.isin([4, 5, 6])

    # Major logic: worldcup always major-like
    e["is_worldcup"] = e["event_type"].astype(str).str.lower().eq("worldcup")
    e["is_major"] = e["event_name"].astype(str).apply(_is_major_event) | e["is_worldcup"]

    # Multipliers (demand-related components)
    e["major_boost"] = 1.0
    e.loc[e["is_worldcup"], "major_boost"] = 1.60
    e.loc[(~e["is_worldcup"]) & (e["is_major"]), "major_boost"] = 1.45

    e["weekend_boost"] = e["date"].map(_weekend_boost)
    e["season_boost"] = e["date"].map(_season_boost)
    e["type_weight"] = e.apply(lambda r: _type_weight(r.get("holiday_type")), axis=1)

    # Closeness (Databricks-like): favors nearer events
    e["closeness"] = e["days_to_event"].apply(lambda x: math.exp(-float(x) / 30.0))

    # Final score (Demand proxy simplified for offline: closeness × multipliers)
    e["score"] = e["closeness"] * e["major_boost"] * e["type_weight"] * e["weekend_boost"] * e["season_boost"]

    # Publish date logic with lead days rules (Databricks-like)
    e["lead_days_used"] = e.apply(
        lambda r: _lead_days_used(int(base_lead_days), str(r.get("event_type", "")), bool(r["is_major"])),
        axis=1
    )
    e["publish_date_raw"] = e["date"] - pd.to_timedelta(e["lead_days_used"], unit="D")
    e["publish_date"] = e["publish_date_raw"]
    e.loc[e["publish_date"] < today_dt, "publish_date"] = today_dt

    # Campaign window
    e["campaign_start"] = e["publish_date"]
    e["campaign_peak"] = e["date"]
    e["campaign_end"] = e["date"] + pd.Timedelta(days=2)

    # Pricing uplift = f(score) × quality_factor, capped by event type
    e["cap_pct"] = e.apply(lambda r: _caps_by_event_type(str(r.get("event_type", "")), bool(r["is_major"])), axis=1)
    e["uplift_pct"] = e.apply(lambda r: _uplift_from_score(float(r["score"]), float(r["cap_pct"]), float(quality_factor)), axis=1)

    if base_price_used is not None:
        e["recommended_price"] = (base_price_used * (1.0 + e["uplift_pct"] / 100.0)).round(2)
    else:
        e["recommended_price"] = None

    # Why
    e["rating_value"] = rating_value
    e["why"] = e.apply(_why_tags, axis=1)

    ranked = e.sort_values(["score", "date"], ascending=[False, True]).reset_index(drop=True)
    best = ranked.iloc[0]

    def _pack(r: pd.Series) -> Dict[str, Any]:
        return {
            "event_type": str(r.get("event_type", "event")).title(),
            "event": str(r["event_name"]),
            "event_date": r["date"].date(),
            "publish_date": r["publish_date"].date(),
            "campaign_start": r["campaign_start"].date(),
            "campaign_peak": r["campaign_peak"].date(),
            "campaign_end": r["campaign_end"].date(),
            "score": float(r["score"]),
            "why": str(r.get("why", "")),
            "uplift_pct": float(r["uplift_pct"]),
            "cap_pct": float(r["cap_pct"]),
            "base_price": float(base_price_used) if base_price_used is not None else None,
            "recommended_price": float(r["recommended_price"]) if base_price_used is not None else None,
            "rating_value": float(rating_value) if rating_value is not None else None,
            "quality_factor": float(quality_factor),
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
