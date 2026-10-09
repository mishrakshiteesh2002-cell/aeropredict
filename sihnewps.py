"""
AeroPredict - Predictive Maintenance & Fleet Availability (SIH26249 prototype)
Run with:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

st.set_page_config(page_title="AeroPredict", page_icon="✈️", layout="wide")
# Custom CSS for modern card styling and layout polish
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e2530;
        border: 1px solid #2d3748;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a202c;
        border-radius: 6px;
        padding: 10px 20px;
        color: #e2e8f0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2b6cb0 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)
# ----------------------------------------------------------------- settings
SENS = {  # sensor id -> friendly name (NASA C-MAPSS FD001 informative sensors)
    "s2": "LPC outlet temp", "s3": "HPC outlet temp", "s4": "LPT outlet temp",
    "s7": "HPC outlet pressure", "s11": "Static pressure", "s12": "Fuel flow ratio",
    "s15": "Bypass ratio", "s17": "Bleed enthalpy",
}
# base value, total drift, direction (used only when the real NASA file is missing)
SYN = {"s2": (642, 2.5, 1), "s3": (1590, 30, 1), "s4": (1408, 40, 1), "s7": (554, 3, -1),
       "s11": (47.5, 1.2, 1), "s12": (522, 3, -1), "s15": (8.4, 0.2, 1), "s17": (392, 6, 1)}
FEATS = ["cycle"] + list(SENS) + [s + "_r" for s in SENS]
ISO_FEATS = [s + "_r" for s in SENS]
RED_T, AMBER_T, RUL_CAP = 30, 60, 125
FLEET_N = 15
CYCLES_PER_DAY = 2
PARTS = ["HPC blade set", "LPT turbine blade", "Fuel pump", "Oil pump seal", "Main bearing"]
BASES = ["Base A", "Base B", "Base C", "Base D"]
COLORS = {"GREEN": "#2e9e4f", "AMBER": "#f0a020", "RED": "#d93636"}
ICON = {"GREEN": "🟢", "AMBER": "🟠", "RED": "🔴"}


# --------------------------------------------------------------------- data
def make_synthetic(n_units=100, seed=42):
    rng = np.random.default_rng(seed)
    frames = []
    for u in range(1, n_units + 1):
        life = int(rng.integers(128, 363))
        k = rng.uniform(0.8, 1.2)
        c = np.arange(1, life + 1)
        prog = (c / life) ** 2.5
        d = {"unit": u, "cycle": c}
        for s, (base, amp, sign) in SYN.items():
            d[s] = base + sign * amp * k * prog + rng.normal(0, amp * 0.08, life) + rng.normal(0, amp * 0.05)
        frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True)


@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_FD001.txt")
    if os.path.exists(path):
        cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
        df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
        source = "Real NASA C-MAPSS FD001 data"
    else:
        df = make_synthetic()
        source = "Synthetic demo data that mimics NASA C-MAPSS (put train_FD001.txt next to app.py to use the real data)"
    df = df[["unit", "cycle"] + list(SENS)].copy()
    df["RUL"] = (df.groupby("unit")["cycle"].transform("max") - df["cycle"]).clip(upper=RUL_CAP)
    roll = df.groupby("unit")[list(SENS)].transform(lambda x: x.rolling(5, min_periods=1).mean())
    for s in SENS:
        df[s + "_r"] = roll[s]
    return df, source


@st.cache_resource
def train_models(df):
    train, test = df[df.unit <= 80], df[df.unit > 80]
    if HAS_XGB:
        model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=0)
    else:
        model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
    model.fit(train[FEATS], train["RUL"])
    pred = np.clip(model.predict(test[FEATS]), 0, RUL_CAP)
    rmse = float(np.sqrt(mean_squared_error(test["RUL"], pred)))
    iso = IsolationForest(contamination=0.02, random_state=0).fit(train[train.cycle <= 60][ISO_FEATS])
    return model, iso, rmse


def predict_rul(model, X):
    return np.clip(model.predict(X[FEATS]), 0, RUL_CAP)


def status_of(rul):
    return "RED" if rul <= RED_T else ("AMBER" if rul <= AMBER_T else "GREEN")


# -------------------------------------------------------------------- fleet
def fleet_snapshot(df, model, iso, days):
    rng = np.random.default_rng(7)
    rows, meta = [], []
    for i, u in enumerate(range(81, 81 + FLEET_N)):
        e = df[df.unit == u]
        life = int(e.cycle.max())
        start = max(10, life - int(rng.integers(25, 170)))
        cyc = int(min(start + days * CYCLES_PER_DAY, life - 1))
        rows.append(e[e.cycle == cyc].iloc[0])
        meta.append((f"Jet-{i + 1:02d}", u, cyc))
    X = pd.DataFrame(rows).reset_index(drop=True)
    rul = predict_rul(model, X)
    odd = iso.decision_function(X[ISO_FEATS]) < -0.16  # stricter than default, fewer false alarms
    fleet = pd.DataFrame({
        "Aircraft": [m[0] for m in meta], "Engine": [m[1] for m in meta], "Cycle": [m[2] for m in meta],
        "Pred RUL": rul.round().astype(int),
        "Status": [status_of(r) for r in rul],
        "Early warning": np.where(odd, "Yes", "No"),
        "Critical part": [PARTS[i % len(PARTS)] for i in range(FLEET_N)],
        "Base": [BASES[i % len(BASES)] for i in range(FLEET_N)],
    })
    return fleet


def ready_pct(fleet, slots, parts):
    n_red = int((fleet.Status == "RED").sum())
    fixed = min(slots, parts, n_red)
    return (len(fleet) - n_red + fixed) / len(fleet) * 100


def make_plan(fleet, slots, parts):
    fleet = fleet.copy()
    red = fleet[fleet.Status == "RED"].sort_values("Pred RUL")
    n_fix = min(slots, parts, len(red))
    fix_ids = set(red.head(n_fix).Aircraft)
    plan = []
    for _, r in fleet.iterrows():
        if r.Status == "RED":
            plan.append("Service now (slot + part ready)" if r.Aircraft in fix_ids else "WAITING: no slot or part")
        elif r.Status == "AMBER":
            plan.append("Pre-order part")
        else:
            plan.append("OK, keep flying")
    fleet["Plan"] = plan
    return fleet


# ---------------------------------------------------------------------- UI
df, source = load_data()
model, iso, rmse = train_models(df)

with st.sidebar:
    st.title("✈️ AeroPredict")
    st.caption("Control panel")
    days = st.slider("Days of operation from today", 0, 40, value=0)
    slots = st.slider("Workshop slots available", 0, 10, value=3)
    parts = st.slider("Spare parts in stock", 0, 12, value=4)
    st.info("Move the sliders and watch the fleet health, alerts and readiness change.")

fleet = make_plan(fleet_snapshot(df, model, iso, days), slots, parts)
ready = ready_pct(fleet, slots, parts)
n = fleet.Status.value_counts()

st.title("✈️ AeroPredict: Predictive Maintenance & Fleet Availability")
st.caption("Know it before it breaks. SIH26249 | Team SANKALP")
st.caption("Data: " + source)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Fleet readiness", f"{ready:.0f}%")
m2.metric("🟢 Healthy", int(n.get("GREEN", 0)))
m3.metric("🟠 Watch", int(n.get("AMBER", 0)))
m4.metric("🔴 Needs repair", int(n.get("RED", 0)))
m5.metric("Model error (RMSE)", f"{rmse:.1f} cycles")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🛩️ Fleet overview", "🔍 Aircraft detail", "📦 Alerts & spare parts", "🎛️ What-if & model", "💰Financials"])

# ---- Tab 1: fleet overview
with tab1:
    st.subheader("Fleet health at a glance")
    show = fleet.copy()
    show["Status"] = show["Status"].map(lambda s: f"{ICON[s]} {s}")
    st.dataframe(show[["Aircraft", "Status", "Pred RUL", "Early warning", "Critical part", "Base", "Plan"]],
                 use_container_width=True, hide_index=True)
    fig = px.bar(fleet.sort_values("Pred RUL"), x="Aircraft", y="Pred RUL", color="Status",
                 color_discrete_map=COLORS, title="Remaining useful life per aircraft (flight cycles)")
    fig.add_hline(y=RED_T, line_dash="dash", line_color="red")
    fig.add_hline(y=AMBER_T, line_dash="dash", line_color="orange")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Red = {RED_T} cycles or less left, Amber = {AMBER_T} or less. RUL = Remaining Useful Life.")

# ---- Tab 2: aircraft detail
with tab2:
    sel = st.selectbox("Choose an aircraft", fleet.Aircraft.tolist())
    r = fleet[fleet.Aircraft == sel].iloc[0]
    e = df[(df.unit == r["Engine"]) & (df.cycle <= r["Cycle"])]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", f"{ICON[r.Status]} {r.Status}")
    c2.metric("Predicted life left", f"{r['Pred RUL']} cycles")
    c3.metric("Flight cycles flown", int(r["Cycle"]))
    c4.metric("Early-warning alert", r["Early warning"])

    chosen = st.multiselect("Sensors to show", list(SENS), default=["s3", "s4", "s7"],
                            format_func=lambda s: f"{s} - {SENS[s]}")
    fig = go.Figure()
    for s in chosen:
        base = e[s].iloc[:10].mean()
        fig.add_scatter(x=e.cycle, y=(e[s] - base) / abs(base) * 100, mode="lines", name=SENS[s])
    fig.update_layout(title="Sensor drift vs healthy baseline (%)", xaxis_title="Flight cycle", yaxis_title="% change")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_scatter(x=e.cycle, y=e["RUL"], mode="lines", name="Actual life left")
    fig2.add_scatter(x=e.cycle, y=predict_rul(model, e), mode="lines", name="Predicted life left")
    fig2.add_hline(y=RED_T, line_dash="dash", line_color="red")
    fig2.update_layout(title="Predicted vs actual remaining life", xaxis_title="Flight cycle", yaxis_title="Cycles left")
    st.plotly_chart(fig2, use_container_width=True)

# ---- Tab 3: alerts and spare parts
with tab3:
    st.subheader("Automatic alerts to the supply chain")
    hot = fleet[fleet.Status != "GREEN"].sort_values("Pred RUL")
    if hot.empty:
        st.success("No alerts. All aircraft are healthy.")
    for _, a in hot.iterrows():
        if a.Status == "RED":
            st.error(f"🔴 ORDER NOW: {a['Critical part']} for {a.Aircraft} at {a.Base}. About {a['Pred RUL']} cycles left. {a.Plan}.")
        else:
            st.warning(f"🟠 PRE-ORDER: {a['Critical part']} for {a.Aircraft} at {a.Base}. About {a['Pred RUL']} cycles left.")
    need_now, need_soon = int((fleet.Status == "RED").sum()), int((fleet.Status == "AMBER").sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Parts needed now", need_now)
    k2.metric("Parts to pre-order", need_soon)
    k3.metric("In stock", parts)
    k4.metric("Shortfall right now", max(0, need_now - parts))
    with st.expander("Historical failure log (from training data)"):
        rng = np.random.default_rng(3)
        log = df[df.unit <= 80].groupby("unit")["cycle"].max().reset_index()
        log.columns = ["Engine", "Cycles flown at failure"]
        log["Failed part"] = [PARTS[int(x)] for x in rng.integers(0, len(PARTS), len(log))]
        st.dataframe(log, use_container_width=True, hide_index=True)
        st.caption("Failed-part labels are illustrative for the demo.")

# ---- Tab 4: what-if and model
with tab4:
    st.subheader("What if we had fewer spare parts or workshop slots?")
    st.write(f"Today: **{slots} slots**, **{parts} parts** gives **{ready:.0f}%** readiness.")
    a_parts = pd.DataFrame({"Spare parts in stock": list(range(0, 13))})
    a_parts["Readiness %"] = [ready_pct(fleet, slots, p) for p in a_parts["Spare parts in stock"]]
    a_slots = pd.DataFrame({"Workshop slots": list(range(0, 11))})
    a_slots["Readiness %"] = [ready_pct(fleet, s, parts) for s in a_slots["Workshop slots"]]
    w1, w2 = st.columns(2)
    w1.plotly_chart(px.line(a_parts, x="Spare parts in stock", y="Readiness %", markers=True,
                            title="Readiness vs spare parts"), use_container_width=True)
    w2.plotly_chart(px.line(a_slots, x="Workshop slots", y="Readiness %", markers=True,
                            title="Readiness vs workshop slots"), use_container_width=True)
    st.subheader("About the model")
    st.write(f"{'XGBoost' if HAS_XGB else 'Gradient Boosting'} predicts remaining life from sensor readings. "
             f"Isolation Forest gives the early-warning flag. Test error: **{rmse:.1f} cycles** (RMSE) on engines the model never saw.")
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=FEATS).nlargest(8).reset_index()
        imp.columns = ["Feature", "Importance"]
        st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h",
                               title="What the model looks at most"), use_container_width=True)

        # ---- Tab 5: Financial Impact & Downtime Savings
with tab5:
    st.subheader("💰 Financial Impact & Downtime Savings Estimator")
    st.write("Translating predictive maintenance into business value for fleet operations.")
    
    # Financial estimation inputs
    c1, c2, c3 = st.columns(3)
    avg_downtime_cost = c1.number_input("Cost per hour of unplanned downtime ($)", value=12000, step=1000)
    repair_cost = c2.number_input("Scheduled maintenance cost ($)", value=35000, step=5000)
    emergency_cost = c3.number_input("Emergency replacement cost ($)", value=95000, step=5000)
    
    red_count = int((fleet.Status == "RED").sum())
    potential_savings = red_count * (emergency_cost - repair_cost)
    
    st.markdown("---")
    f1, f2 = st.columns(2)
    f1.metric("Estimated Unplanned Failures Avoided", f"{red_count} Aircraft")
    f2.metric("Projected Cost Savings", f"${potential_savings:,.0f}", delta="Optimized Maintenance")
    
    st.info("💡 **Hackathon Pitch Tip:** Showing clear financial ROI alongside technical accuracy proves both the engineering viability and business impact of AeroPredict.")