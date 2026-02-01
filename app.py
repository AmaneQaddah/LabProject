# app.py
from datetime import date
import pandas as pd
import streamlit as st

from recommender import recommend_publish_date

st.set_page_config(page_title="HOSTBOOST", page_icon="🚀", layout="centered")

DEFAULT_BASE_LEAD_DAYS = 14   # hidden
DEFAULT_TOP_K_BY_TIME = 60    # hidden (take nearest events first)

st.title("🚀 HOSTBOOST")
st.caption("Choose a listing and get the best event-based publish date + a bounded pricing uplift based on demand (score) and quality (rating).")

property_id = st.text_input("Property ID", placeholder="Paste your property_id (exactly as in the dataset)")
today = st.date_input("Today", value=date.today())

base_price = st.number_input(
    "Current nightly price (optional)",
    min_value=0.0, value=0.0, step=1.0
)

with st.expander("⚙️ Recommendation settings (optional)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        lookahead_days = st.selectbox("Lookahead window (days)", [90, 180, 365, 730], index=2)
    with col2:
        alternatives_n = st.selectbox("Alternatives to show", [0, 1, 2, 3, 4, 5], index=3)

run = st.button("Recommend", type="primary")


@st.cache_data(show_spinner=False)
def _cached_recommend(pid: str, today_val: date, lookahead: int, alts: int, base_price_val: float):
    return recommend_publish_date(
        property_id=pid,
        today=today_val,
        base_lead_days=DEFAULT_BASE_LEAD_DAYS,
        lookahead_days=lookahead,
        top_k_by_time=DEFAULT_TOP_K_BY_TIME,
        alternatives_n=alts,
        base_price=(base_price_val if base_price_val > 0 else None),
    )


def _pretty_country(c: str) -> str:
    c = (c or "").strip()
    if not c:
        return ""
    return c.replace("-", " ").title()


if run:
    pid = property_id.strip()
    if not pid:
        st.error("Please enter a Property ID.")
        st.stop()

    result, err = _cached_recommend(pid, today, int(lookahead_days), int(alternatives_n), float(base_price))
    if err:
        st.error(err)
        st.stop()

    best = result["best"]
    country = _pretty_country(result.get("country", ""))

    st.success("Recommendation found!")

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Country", country if country else "N/A")
    m2.metric("Event type", str(best.get("event_type", "")).title())
    m3.metric("Publish date", best["publish_date"].strftime("%m-%d"))
    m4.metric("Score", f"{best['score']:.4f}")

    # Pricing + quality metrics
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Uplift (%)", f"{best['uplift_pct']:.1f}%")
    p2.metric("Base price", f"{best['base_price']:.2f}" if best.get("base_price") is not None else "N/A")
    p3.metric("Recommended price", f"{best['recommended_price']:.2f}" if best.get("recommended_price") is not None else "N/A")
    p4.metric("Rating", f"{best['rating_value']:.2f}" if best.get("rating_value") is not None else "N/A")

    with st.container(border=True):
        st.subheader("✅ Best Recommendation")
        st.markdown(f"**Event:** {best['event']}")
        st.markdown(f"**Event date:** {best['event_date'].strftime('%Y-%m-%d')}")
        st.markdown(f"**Recommended publish date:** 🎯 **{best['publish_date'].strftime('%Y-%m-%d')}**")

        st.divider()
        st.markdown("**Pricing recommendation:**")
        st.write(f"Increase by **{best['uplift_pct']:.1f}%**")
        if best.get("recommended_price") is not None and best.get("base_price") is not None:
            st.write(f"→ Suggested nightly price: **{best['recommended_price']:.2f}** (base: {best['base_price']:.2f})")

        if best.get("quality_factor") is not None:
            st.caption(f"Quality factor (from rating): ×{best['quality_factor']:.2f}")

        st.divider()
        st.markdown("**Campaign window:**")
        c1, c2, c3 = st.columns(3)
        c1.write(f"Start: **{best['campaign_start'].strftime('%Y-%m-%d')}**")
        c2.write(f"Peak: **{best['campaign_peak'].strftime('%Y-%m-%d')}**")
        c3.write(f"End: **{best['campaign_end'].strftime('%Y-%m-%d')}**")

        if best.get("why"):
            st.info(f"Why: {best['why']}")

    # Alternatives table
    alts = result.get("alternatives", [])
    if alts:
        st.subheader("✨ Top alternatives")
        df = pd.DataFrame(alts).copy()
        df["event_date"] = df["event_date"].apply(lambda d: d.strftime("%Y-%m-%d"))
        df["publish_date"] = df["publish_date"].apply(lambda d: d.strftime("%Y-%m-%d"))
        df["uplift_pct"] = df["uplift_pct"].apply(lambda x: round(float(x), 1))
        df["score"] = df["score"].apply(lambda x: round(float(x), 4))

        show_cols = ["event_type", "event", "event_date", "publish_date", "score", "uplift_pct", "rating_value"]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True)
