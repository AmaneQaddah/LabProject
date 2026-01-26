# app.py
from datetime import date
import pandas as pd
import streamlit as st

from recommender import recommend_publish_date

st.set_page_config(page_title="Ad Publish Date Recommender", page_icon="📢", layout="centered")

DEFAULT_BASE_LEAD_DAYS = 14  # hidden
DEFAULT_TOP_K_BY_TIME = 60   # hidden

st.title("📢 Ad Publish Date Recommender  🏡✨")
st.caption("Pick a property, and we’ll rank upcoming holidays + events and recommend publish date + pricing uplift (%)")

property_id = st.text_input("Property ID", placeholder="Paste your property_id (exactly as in the dataset)")
today = st.date_input("Today", value=date.today())

base_price = st.number_input(
    "Current nightly price (optional)",
    min_value=0.0, value=0.0, step=1.0
)

with st.expander("⚙️ Recommendation settings (optional)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        lookahead_days = st.selectbox("Lookahead window", [90, 180, 365, 730], index=2)
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
    country = result["country"].title()
    st.success("Recommendation found!")

    # Top metrics (NO lead days)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Country", country)
    m2.metric("Event type", str(best.get("event_type", "")).title())
    m4.metric("Score", f"{best['score']:.4f}")

    # Pricing metrics
    p1, p2, p3 = st.columns(3)
    p1.metric("Uplift (%)", f"{best['price_uplift_pct']:.1f}%")
    p2.metric("Base price", f"{best['base_price_used']:.2f}" if best.get("base_price_used") is not None else "N/A")
    p3.metric("Recommended price", f"{best['recommended_price']:.2f}" if best.get("recommended_price") is not None else "N/A")

    with st.container(border=True):
        st.subheader("✅ Best Recommendation")
        st.markdown(f"**Event:** {best['event']}")
        st.markdown(f"**Event date:** {best['event_date'].strftime('%Y-%m-%d')}")
        st.markdown(f"**Recommended publish date:** 🎯 **{best['publish_date'].strftime('%Y-%m-%d')}**")

        st.divider()
        st.markdown("**Pricing recommendation:**")
        st.write(f"Increase by **{best['price_uplift_pct']:.1f}%**")
        if best.get("recommended_price") is not None:
            st.write(f"→ Suggested nightly price: **{best['recommended_price']:.2f}** (base: {best['base_price_used']:.2f})")

        st.divider()
        st.markdown("**Campaign window:**")
        c1, c2, c3 = st.columns(3)
        c1.write(f"Start: **{best['campaign_start'].strftime('%Y-%m-%d')}**")
        c2.write(f"Peak: **{best['campaign_peak'].strftime('%Y-%m-%d')}**")
        c3.write(f"End: **{best['campaign_end'].strftime('%Y-%m-%d')}**")

        if best.get("why"):
            st.info(f"Why: {best['why']}")

    alts = result.get("alternatives", [])
    if alts:
        st.subheader("✨ Top alternatives")
        df = pd.DataFrame(alts)
        df["event_date"] = df["event_date"].apply(lambda d: d.strftime("%Y-%m-%d"))
        df["publish_date"] = df["publish_date"].apply(lambda d: d.strftime("%Y-%m-%d"))

        cols = ["event_type", "event", "event_date", "publish_date"]
        st.dataframe(df[cols], use_container_width=True)
