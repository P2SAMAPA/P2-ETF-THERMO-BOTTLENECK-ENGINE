import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Thermo-Bottleneck", layout="wide")
st.title("🔥 Thermo-Bottleneck – Loss of Order Detection")
st.caption("Information Bottleneck | Transfer Entropy | MEP rate | Trend Collapse Warning")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'thermo_bottleneck' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error loading JSON: {data['error']}")
    st.stop()

st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"**Run date:** {data['run_date']}")
st.sidebar.write(f"**Next trading day:** {next_trading_day()}")
st.sidebar.write("**Method:** VIB compression + TE macro→ETF + MEP rate → Collapse Score")

universes = data["universes"]
if not universes:
    st.warning("No universe data.")
    st.stop()

st.header("⚠️ Trend Collapse Warning")

for universe_name, uni_data in universes.items():
    warning = uni_data.get("collapse_warning", False)
    avg_score = uni_data.get("average_collapse_score", 0.0)
    threshold = uni_data.get("threshold", 1.0)
    win = uni_data.get("selected_window", "?")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"{universe_name}", "🚨 COLLAPSE RISK" if warning else "✅ STABLE", delta=None)
    with col2:
        st.metric("Avg Collapse Score", f"{avg_score:.3f}")
    with col3:
        st.metric("Window (days)", win)
    
    # Risk ETFs
    risk_etfs = uni_data.get("collapse_risk_etfs", [])
    if risk_etfs:
        st.write("**ETFs with highest collapse score (most loss of order):**")
        for r in risk_etfs[:3]:
            st.markdown(f"- **{r['ticker']}** (score {r['collapse_score']:.3f})")
    # Explanation
    with st.expander("📘 What does 'Loss of Order' mean?"):
        st.markdown("""
        - **Loss of order** = increasing randomness / decreasing predictability.
        - Detected via:
          - **Information Bottleneck (IB) compression**: falling compression efficiency means the past no longer explains the future.
          - **Transfer Entropy (TE) from macro to ETF**: decreasing TE means external drivers are lost, system becomes self‑driven.
          - **Maximum Entropy Production (MEP) rate**: rising MEP indicates the system is far from equilibrium.
        - **Collapse Score** = -IB - TE + MEP. High score → imminent trend collapse.
        """)
    st.divider()

# Detailed view
universe_names = list(universes.keys())
selected = st.selectbox("Select Universe for detailed view", universe_names)

if selected:
    uni_data = universes[selected]
    all_tickers = uni_data.get("all_tickers", {})
    if all_tickers:
        rows = []
        for ticker, metrics in all_tickers.items():
            rows.append({
                "ETF": ticker,
                "Collapse Score": metrics["collapse_score"],
                "IB Compression": metrics["ib_compression"],
                "TE (macro→ETF)": metrics["transfer_entropy_macro"],
                "MEP Rate": metrics["mep_rate"]
            })
        df = pd.DataFrame(rows).sort_values("Collapse Score", ascending=False)
        st.subheader("📊 Detailed Metrics per ETF")
        st.dataframe(df, use_container_width=True, hide_index=True)
        fig = px.bar(df, x="ETF", y="Collapse Score", title="Loss of Order Score per ETF (higher = higher collapse risk)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No per‑ticker data available.")

st.caption("Collapse warning is active when the average score exceeds the threshold. The engine picks the rolling window that best predicts future drawdown.")
