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
HOLIDAYS_PATH = os.getenv("HOLIDAYS_CSV_PATH", "data/holidays_sample_100.csv")

# =========================
# Major keywords (token-style)
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

    # High impact
    if "public holiday" in ht: return 1.00
    if "national holiday" in ht: return 1.00
    if "statutory holiday" in ht: return 1.00
    if "official holiday" in ht: return 0.95
    if "federal holiday" in ht: return 0.95

    # Medium
    if "bank holiday" in ht: return 0.90
    if "government holiday" in ht: return 0.90
    if "regional government" in ht: return 0.85
    if "state holiday" in ht: return 0.85

    # Lower
    if "half-day" in ht: return 0.75
    if "special" in ht: return 0.75
    if "silent day" in ht: return 0.70

    return 0.75


def _lead_days_used(base_lead: int, is_major: bool) -> int:
    # used only for publish date logic (not for score)
    return max(base_lead, 21) if is_major else base_lead


def _why_tags(row: pd.Series) -> str:
    tags: List[str] = []
    if row["is_major"]:
        tags.append("Major holiday")
    if row["is_weekend"]:
        tags.append("Weekend")
    tags.append(f"In {int(row['days_to_event'])} days")
    tags.append(f"Score={row['score']:.3f}")
    if pd.notna(row.get("holiday_type")) and str(row.get("holiday_type")).strip():
        tags.append(f"Type: {row['holiday_type']}")
    return ", ".join(tags)


# =========================
# Rating / quality helpers (stable, low-risk)
# =========================
RATING_CANDIDATES = [
    "rating", "ratings", "review_rating", "avg_rating", "overall_rating",
    "category_rating"
]


def _extract_first_number_from_any(v) -> Optional[float]:
    try:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        # keep digits and dot only (first number)
        num = ""
        seen = False
        for ch in s:
            if ch.isdigit() or (ch == "." and seen):
                num += ch
            elif ch.isdigit():
                seen = True
                num += ch
            elif seen:
                break
        x = float(num) if num else None
        if x is None or math.isnan(x) or x <= 0:
            return None
        return x
    except Exception:
        return None


def _get_rating_value(airbnb_row: pd.Series) -> Optional[float]:
    for col in RATING_CANDIDATES:
        if col in airbnb_row.index:
            x = _extract_first_number_from_any(airbnb_row.get(col))
            if x is not None:
                return x
    return None


def _quality_factor_from_rating(rating_value: Optional[float]) -> float:
    """
    Map rating -> multiplicative factor in a tight range (doesn't explode).
    Assumption: rating on a 0..5 scale. Missing -> neutral.
    """
    if rating_value is None:
        return 1.00  # neutral if missing
    r = float(rating_value)
    # guard rails for odd scales
    if r > 5.0 and r <= 100.0:
        r = (r / 100.0) * 5.0
    r = max(0.0, min(r, 5.0))
    r_norm = r / 5.0  # 0..1
    # 0.90 .. 1.10 (tight)
    return 0.90 + 0.20 * r_norm


# =========================
# Pricing uplift heuristic
# - Score controls uplift monotonically
# - days_to_event affects ONLY urgency (small add)
# - quality factor scales uplift (safe)
# =========================
def _price_uplift_pct(score: float, days_to_event: int, is_major: bool, is_weekend: bool,
                     holiday_type: Optional[str], quality_factor: float) -> float:
    # monotonic mapping from score -> 0..1
    score_norm = 1.0 - math.exp(-max(score, 0.0) / 1.2)  # 0..1

    # base uplift from score (monotonic)
    pct = score_norm * 22.0

    # add small boosts (still monotonic w.r.t score for a fixed listing)
    if is_major:
        pct += 6.0
    if is_weekend:
        pct += 4.0

    # urgency pricing only
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

    # apply quality factor (tight range)
    pct = pct * float(quality_factor)

    # hard cap (stable)
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
    # extended to support typical cleaned columns
    for col in ["nightly_price", "price_clean", "total_price_clean", "price", "base_price", "avg_price"]:
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
def _load_holidays() -> pd.DataFrame:
    if not os.path.exists(HOLIDAYS_PATH):
        raise FileNotFoundError(f"Missing file: {HOLIDAYS_PATH}")

    df = pd.read_csv(HOLIDAYS_PATH)
    if "country_mapped" not in df.columns:
        raise ValueError("Holidays CSV must include column: country_mapped")

    df["country_mapped"] = df["country_mapped"].astype(str).str.strip().str.lower()

    date_col = "holiday_date" if "holiday_date" in df.columns else ("date" if "date" in df.columns else None)
    if date_col is None:
        raise ValueError("Holidays CSV must have 'holiday_date' or 'date' column")

    if "holiday_name" not in df.columns and "name" in df.columns:
        df["holiday_name"] = df["name"].astype(str)
    if "holiday_name" not in df.columns:
        df["holiday_name"] = "Holiday"

    if "holiday_type" not in df.columns:
        df["holiday_type"] = None

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df["date"].notna()].copy()

    df["event_name"] = df["holiday_name"].astype(str)
    df["event_type"] = "holiday"
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
        events = _load_holidays()
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

    # rating / quality factor
    rating_value = _get_rating_value(airbnb_row)
    quality_factor = _quality_factor_from_rating(rating_value)

    today_dt = pd.to_datetime(today).normalize()
    max_dt = today_dt + pd.Timedelta(days=int(lookahead_days))

    e = events[events["country_mapped"] == country].copy()
    if e.empty:
        return None, "No holidays for this country"

    e = e[(e["date"] >= today_dt) & (e["date"] <= max_dt)].copy()
    if e.empty:
        return None, "No upcoming holidays in the selected lookahead window"

    # keep only nearest N events by time (speed & stable)
    e = e.sort_values("date").head(int(top_k_by_time)).copy()

    e["days_to_event"] = (e["date"] - today_dt).dt.days
    e["is_major"] = e["event_name"].astype(str).apply(_is_major_event)
    e["is_weekend"] = e["date"].dt.weekday.isin([4, 5, 6])

    e["major_boost"] = 1.0
    e.loc[e["is_major"], "major_boost"] = 1.45

    e["weekend_boost"] = e["date"].map(_weekend_boost)
    e["season_boost"] = e["date"].map(_season_boost)
    e["type_weight"] = e.apply(lambda r: _type_weight(r.get("holiday_type")), axis=1)

    # ✅ SCORE WITHOUT TIME EFFECT
    e["score"] = e["major_boost"] * e["type_weight"] * e["weekend_boost"] * e["season_boost"]

    # publish date logic
    e["lead_days_used"] = e.apply(
        lambda r: _lead_days_used(int(base_lead_days), bool(r["is_major"])),
        axis=1
    )
    e["publish_date_raw"] = e["date"] - pd.to_timedelta(e["lead_days_used"], unit="D")
    e["publish_date"] = e["publish_date_raw"]
    e.loc[e["publish_date"] < today_dt, "publish_date"] = today_dt

    e["campaign_start"] = e["publish_date"]
    e["campaign_peak"] = e["date"]
    e["campaign_end"] = e["date"] + pd.Timedelta(days=2)

    e["why"] = e.apply(_why_tags, axis=1)

    # pricing uplift: score + urgency + quality (safe)
    e["uplift_pct"] = e.apply(
        lambda r: _price_uplift_pct(
            score=float(r["score"]),
            days_to_event=int(r["days_to_event"]),
            is_major=bool(r["is_major"]),
            is_weekend=bool(r["is_weekend"]),
            holiday_type=r.get("holiday_type"),
            quality_factor=quality_factor,
        ),
        axis=1
    )

    if base_price_used is not None:
        e["recommended_price"] = (base_price_used * (1.0 + e["uplift_pct"] / 100.0)).round(2)
    else:
        e["recommended_price"] = None

    ranked = e.sort_values(["score", "date"], ascending=[False, True]).reset_index(drop=True)
    best = ranked.iloc[0]

    def _pack(r: pd.Series) -> Dict[str, Any]:
        return {
            "event_type": str(r.get("event_type", "holiday")).title(),
            "event": str(r["event_name"]),
            "event_date": r["date"].date(),
            "publish_date": r["publish_date"].date(),
            "score": float(r["score"]),
            "why": str(r["why"]),
            "campaign_start": r["campaign_start"].date(),
            "campaign_peak": r["campaign_peak"].date(),
            "campaign_end": r["campaign_end"].date(),
            "uplift_pct": float(r["uplift_pct"]),
            "base_price": float(base_price_used) if base_price_used is not None else None,
            "recommended_price": float(r["recommended_price"]) if base_price_used is not None else None,
            "rating_value": float(rating_value) if rating_value is not None else None,
            "quality_factor": float(quality_factor),
            # internal
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
