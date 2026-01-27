# recommender.py
import os
import math
from datetime import timedelta
from functools import lru_cache
from typing import Tuple, Dict, Any, Optional, List

import pandas as pd

# =========================
# Data paths (CSV recommended for deployment)
# =========================
AIRBNB_PATH = os.getenv("AIRBNB_CSV_PATH", "data/airbnb_sample_100.csv")
HOLIDAYS_PATH = os.getenv("HOLIDAYS_CSV_PATH", "data/holidays_sample_100.csv")  # או holidays_full.csv

# =========================
# Heuristics
# =========================
MAJOR_KEYWORDS = [
    "christmas", "new year", "easter", "ramadan", "eid", "diwali", "carnival",
    "thanksgiving", "independence", "national day", "labor day", "may day",
    "hannukah", "hanukkah", "passover", "holi", "vesak", "songkran",
    "rosh", "kippur", "sukkot", "shavuot", "purim"
]

TYPE_WEIGHT = {
    "public": 1.00,
    "bank": 0.90,
    "school": 0.85,
    "religious": 0.85,
    "optional": 0.70,
    "observance": 0.65
}

def _is_major(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in MAJOR_KEYWORDS)

def _get_type_weight(row: pd.Series) -> float:
    t = ""
    for col in ["holiday_type", "type", "category"]:
        if col in row and pd.notna(row[col]):
            t = str(row[col]).strip().lower()
            break
    return TYPE_WEIGHT.get(t, 0.75)

def _weekend_multiplier(d) -> float:
    wd = d.weekday()  # Mon=0..Sun=6
    return 1.18 if wd in (4, 5, 6) else 1.00

def _season_multiplier(d) -> float:
    m = d.month
    if m in (6, 7, 8):
        return 1.08
    if m in (11, 12):
        return 1.06
    return 1.00

def _holiday_score(row: pd.Series, today_date) -> float:
    """
    Score WITHOUT World Cup.
    (שימי לב: כאן עדיין יש השפעה עקיפה של "עבר/עתיד" רק דרך סינון בהמשך.)
    """
    d = row["date"]
    days = (d - today_date).days
    if days < 0:
        return -1e9

    name = str(row.get("holiday_name", "") or row.get("name", "") or "")
    major_boost = 1.45 if _is_major(name) else 1.00
    type_w = _get_type_weight(row)
    weekend_boost = _weekend_multiplier(d)
    season_boost = _season_multiplier(d)

    # בלי closeness (אם תרצי להחזיר closeness תגידי)
    return major_boost * type_w * weekend_boost * season_boost

def _dynamic_lead_days(holiday_name: str, base_lead: int) -> int:
    return max(base_lead, 21) if _is_major(holiday_name) else base_lead

def _reason_tags(row: pd.Series, today_date) -> List[str]:
    tags = []
    name = str(row.get("holiday_name", "") or "")
    d = row["date"]
    days = (d - today_date).days

    if _is_major(name):
        tags.append("Major holiday")
    if d.weekday() in (4, 5, 6):
        tags.append("Weekend boost")
    for col in ["holiday_type", "type", "category"]:
        if col in row and pd.notna(row[col]):
            tags.append(f"Type: {str(row[col])}")
            break
    if days <= 30:
        tags.append("Within 30 days")
    return tags

# =========================
# Cached loaders (important for Streamlit speed)
# =========================
@lru_cache(maxsize=1)
def _load_airbnb() -> pd.DataFrame:
    df = pd.read_csv(AIRBNB_PATH)
    if "property_id" not in df.columns or "country_mapped" not in df.columns:
        raise ValueError("Airbnb CSV must include columns: property_id, country_mapped")
    df["property_id"] = df["property_id"].astype(str).str.strip()
    df["country_mapped"] = df["country_mapped"].astype(str).str.strip().str.lower()
    return df

@lru_cache(maxsize=1)
def _load_holidays() -> pd.DataFrame:
    df = pd.read_csv(HOLIDAYS_PATH)
    if "country_mapped" not in df.columns:
        raise ValueError("Holidays CSV must include column: country_mapped")

    df["country_mapped"] = df["country_mapped"].astype(str).str.strip().str.lower()

    # detect date col
    date_col = "date" if "date" in df.columns else ("holiday_date" if "holiday_date" in df.columns else None)
    if date_col is None:
        raise ValueError("Holidays CSV must include date column: 'date' or 'holiday_date'")

    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    df = df[df["date"].notna()].copy()

    # ensure holiday_name
    if "holiday_name" not in df.columns:
        if "name" in df.columns:
            df["holiday_name"] = df["name"].astype(str)
        else:
            df["holiday_name"] = "Holiday"

    return df

# =========================
# Main API (same signature your app uses)
# =========================
def recommend_publish_date(
    property_id: str,
    today,
    base_lead_days: int = 14,
    lookahead_days: int = 365,
    top_k_by_time: int = 20,
    alternatives_n: int = 3
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:

    try:
        airbnb = _load_airbnb()
        holidays = _load_holidays()
    except Exception as e:
        return None, f"Dataset load error: {e}"

    pid = str(property_id).strip()
    row = airbnb[airbnb["property_id"] == pid]
    if row.empty:
        return None, "Property not found"

    country = row.iloc[0].get("country_mapped")
    if not country or (isinstance(country, float) and pd.isna(country)):
        return None, "Country mapping missing for this property"

    country = str(country).strip().lower()
    today_date = pd.to_datetime(today).date()

    # filter holidays by country
    h = holidays[holidays["country_mapped"] == country].copy()
    if h.empty:
        return None, "No holidays for this country"

    # future window
    h = h[(h["date"] >= today_date) & (h["date"] <= (today_date + timedelta(days=int(lookahead_days))))].copy()
    if h.empty:
        return None, "No upcoming holidays in the selected lookahead window"

    # limit by soonest N dates for efficiency
    h = h.sort_values("date").head(int(top_k_by_time)).copy()

    # score + rank
    h["score"] = h.apply(lambda r: _holiday_score(r, today_date), axis=1)
    ranked = h.sort_values(["score", "date"], ascending=[False, True]).reset_index(drop=True)

    best = ranked.iloc[0]
    holiday_name = str(best["holiday_name"])
    lead_used = _dynamic_lead_days(holiday_name, int(base_lead_days))

    publish_date = best["date"] - timedelta(days=lead_used)
    if publish_date < today_date:
        publish_date = today_date

    campaign_start = publish_date
    campaign_peak = best["date"]
    campaign_end = best["date"] + timedelta(days=2)

    alternatives = []
    for i in range(1, min(1 + int(alternatives_n), len(ranked))):
        r = ranked.iloc[i]
        alt_name = str(r["holiday_name"])
        alt_lead = _dynamic_lead_days(alt_name, int(base_lead_days))
        alt_publish = r["date"] - timedelta(days=alt_lead)
        if alt_publish < today_date:
            alt_publish = today_date

        alternatives.append({
            "holiday": alt_name,
            "holiday_date": r["date"],
            "publish_date": alt_publish,
            "lead_days_used": alt_lead,
            "score": float(r["score"]),
            "why": ", ".join(_reason_tags(r, today_date))
        })

    return {
        "property_id": pid,
        "country": country,
        "best": {
            "holiday": holiday_name,
            "holiday_date": best["date"],
            "publish_date": publish_date,
            "lead_days_used": lead_used,
            "score": float(best["score"]),
            "why": ", ".join(_reason_tags(best, today_date)),
            "campaign_start": campaign_start,
            "campaign_peak": campaign_peak,
            "campaign_end": campaign_end
        },
        "alternatives": alternatives
    }, None
