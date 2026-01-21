import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Water Quality Monitoring Dashboard",
    layout="wide"
)

# =====================================================
# LIGHT THEME (ENTIRE DASHBOARD)
# =====================================================
st.markdown("""
<style>
.stApp {
    background-color: #f8fafc;
    color: #0f172a;
}

h1, h2, h3, h4 {
    color: #0284c7;
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background-color: #ffffff;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
}

/* Buttons */
button {
    background-color: #0284c7 !important;
    color: white !important;
    border-radius: 8px;
}

/* Metric cards */
.metric-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    text-align: center;
}

/* Alerts */
.stAlert-success { background-color: #dcfce7; }
.stAlert-warning { background-color: #fef9c3; }
.stAlert-error { background-color: #fee2e2; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_water_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "water_quality_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

# =====================================================
# LOAD DATA & MODEL
# =====================================================
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

# =====================================================
# HEADER
# =====================================================
st.markdown("## 💧 Water Quality Monitoring & Risk Prediction")
st.markdown("Analyze water quality using **State, District, Year, and Month filters**")
st.markdown("---")

# =====================================================
# FILTERS
# =====================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    state = st.selectbox("State", sorted(df["state"].unique()))
with c2:
    district = st.selectbox("District", sorted(df[df["state"] == state]["district"].unique()))
with c3:
    year = st.selectbox("Year", sorted(df[(df["state"] == state) & (df["district"] == district)]["year"].unique()))
with c4:
    month = st.selectbox("Month", sorted(df[(df["state"] == state) & (df["district"] == district) & (df["year"] == year)]["month"].unique()))

filtered = df[
    (df["state"] == state) &
    (df["district"] == district) &
    (df["year"] == year) &
    (df["month"] == month)
]

# =====================================================
# SUMMARY
# =====================================================
st.markdown("### 📊 Summary")

def metric_card(title, value):
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color:#64748b;">{title}</h4>
        <h2 style="color:#0f172a;">{value}</h2>
    </div>
    """, unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)

with s1:
    metric_card("Records", len(filtered))
with s2:
    metric_card("Avg pH", round(filtered["pH"].mean(), 2) if not filtered.empty else 0)
with s3:
    metric_card("Avg DO", round(filtered["dissolved_oxygen"].mean(), 2) if not filtered.empty else 0)

# =====================================================
# WATER QUALITY DISTRIBUTION
# =====================================================
st.markdown("### 📊 Water Quality Distribution")

if not filtered.empty:
    dist = filtered["water_status"].value_counts().reset_index()
    dist.columns = ["Status", "Count"]

    fig_dist = px.pie(
        dist,
        names="Status",
        values="Count",
        hole=0.55,
        color="Status",
        color_discrete_map={
            "Safe": "#22c55e",
            "Moderate": "#facc15",
            "Polluted": "#ef4444"
        }
    )

    fig_dist.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a")
    )

    st.plotly_chart(fig_dist, use_container_width=True)
else:
    st.info("No data available")

# =====================================================
# PARAMETER TRENDS
# =====================================================
st.markdown("### 📉 Parameter Trends Over Years")

trend = (
    df[(df["state"] == state) & (df["district"] == district)]
    .groupby("year")
    .mean(numeric_only=True)
    .reset_index()
)

fig_trend = px.line(
    trend,
    x="year",
    y=["pH", "dissolved_oxygen", "bod", "turbidity"],
    markers=True,
    color_discrete_map={
        "pH": "#0284c7",
        "dissolved_oxygen": "#22c55e",
        "bod": "#ef4444",
        "turbidity": "#f59e0b"
    }
)

fig_trend.update_layout(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#0f172a"),
    xaxis=dict(gridcolor="#e5e7eb"),
    yaxis=dict(gridcolor="#e5e7eb")
)

st.plotly_chart(fig_trend, use_container_width=True)

# =====================================================
# TOP STATES
# =====================================================
st.markdown("### 🏆 Top States by Water Quality Records")

top_states = (
    df.groupby(["state", "water_status"])
      .size()
      .reset_index(name="count")
)

fig_top = px.bar(
    top_states,
    x="count",
    y="state",
    color="water_status",
    orientation="h",
    color_discrete_map={
        "Safe": "#22c55e",
        "Moderate": "#facc15",
        "Polluted": "#ef4444"
    }
)

fig_top.update_layout(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#0f172a"),
    xaxis=dict(gridcolor="#e5e7eb"),
    yaxis=dict(gridcolor="#e5e7eb")
)

st.plotly_chart(fig_top, use_container_width=True)

# =====================================================
# ML PREDICTION
# =====================================================
st.markdown("### 🔍 Water Quality Prediction")

if not filtered.empty:
    r = filtered.iloc[0]

    input_df = pd.DataFrame([[
        r["temperature"], r["pH"], r["dissolved_oxygen"],
        r["conductivity"], r["turbidity"], r["bod"],
        r["nitrate"], r["fecal_coliform"]
    ]], columns=[
        "temperature","pH","dissolved_oxygen",
        "conductivity","turbidity","bod",
        "nitrate","fecal_coliform"
    ])

    result = le.inverse_transform(model.predict(input_df))[0]

    if result == "Safe":
        st.success("✅ SAFE")
    elif result == "Moderate":
        st.warning("⚠️ MODERATE")
    else:
        st.error("❌ POLLUTED")
