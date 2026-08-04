"""
Eval trend dashboard. Reads eval_results/history.csv (appended to by
src/eval.py on every run) and visualizes how RAGAS scores and latency
change across pipeline configurations.

Run: streamlit run dashboard/app.py
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
HISTORY_FILE = ROOT / "eval_results" / "history.csv"
SRC_EVAL = ROOT / "src" / "eval.py"

st.set_page_config(
    page_title="RAG Eval Dashboard",
    layout="wide",
    page_icon="📊",
)

st.title("📊 RAG Evaluation & Observability Dashboard")
st.caption("Track how pipeline configurations affect RAGAS scores and latency over time.")

# --------------------------------------------------------
# Run Eval Panel
# --------------------------------------------------------

with st.expander("▶️ Run New Evaluation", expanded=False):
    st.markdown(
        "Trigger an evaluation run against the golden dataset. "
        "Results will be appended to `eval_results/history.csv` and appear below."
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    eval_config = col_a.text_input("Config name (username)", value="default")
    eval_mode = col_b.selectbox("Retrieval mode", ["hybrid", "dense", "sparse"])
    eval_top_k = col_c.slider("Top-K", min_value=1, max_value=10, value=5)
    eval_reranker = col_d.toggle("Use reranker", value=True)

    eval_qe = st.toggle("Use query expansion", value=False)

    if st.button("🚀 Run Eval Now", type="primary"):
        cmd = [
            sys.executable,
            str(SRC_EVAL),
            "--config-name", eval_config,
            "--retrieval-mode", eval_mode,
            "--top-k", str(eval_top_k),
        ]
        if eval_reranker:
            cmd.append("--use-reranker")
        if eval_qe:
            cmd.append("--use-query-expansion")

        with st.spinner("Running evaluation pipeline… this may take a minute."):
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )

        if result.returncode == 0:
            st.success("✅ Eval run complete!")
            st.code(result.stdout, language="text")
            st.rerun()
        else:
            st.error("❌ Eval run failed.")
            st.code(result.stderr or result.stdout, language="text")

st.divider()

# --------------------------------------------------------
# Load History
# --------------------------------------------------------

if not HISTORY_FILE.exists():
    st.warning(
        "No eval runs yet. Use the **▶️ Run New Evaluation** panel above, "
        "or run `python src/eval.py --config-name default --retrieval-mode hybrid --use-reranker` "
        "from the project root."
    )
    st.stop()

df = pd.read_csv(HISTORY_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# --------------------------------------------------------
# Latest Run — KPI Metrics
# --------------------------------------------------------

st.subheader("🏆 Latest Run")
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else None

cols = st.columns(5)
for i, metric in enumerate(METRIC_COLS):
    delta = round(float(latest[metric]) - float(prev[metric]), 3) if prev is not None else None
    cols[i].metric(
        label=metric.replace("_", " ").title(),
        value=f"{latest[metric]:.3f}",
        delta=f"{delta:+.3f}" if delta is not None else None,
    )
cols[4].metric(
    label="Avg Latency (ms)",
    value=f"{latest['avg_latency_ms']:.0f}",
    delta=f"{latest['avg_latency_ms'] - prev['avg_latency_ms']:+.0f} ms" if prev is not None else None,
    delta_color="inverse",
)

st.divider()

# --------------------------------------------------------
# RAGAS Scores Across Pipeline Variants — Bar Chart
# --------------------------------------------------------

st.subheader("📊 RAGAS Scores Across Pipeline Variants")

melted = df.melt(
    id_vars=["run_label", "timestamp"],
    value_vars=METRIC_COLS,
    var_name="metric",
    value_name="score",
)

fig_bar = px.bar(
    melted,
    x="run_label",
    y="score",
    color="metric",
    barmode="group",
    title="Eval Metrics by Configuration",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig_bar.update_layout(
    xaxis_title="Pipeline Configuration",
    yaxis_title="Score (0 – 1)",
    yaxis_range=[0, 1.05],
    legend_title="Metric",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
fig_bar.update_xaxes(tickangle=-20)
st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------------------------------------
# Metric Trend Lines Over Time
# --------------------------------------------------------

st.subheader("📈 Metric Trends Over Time")

fig_trend = go.Figure()
colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

for metric, color in zip(METRIC_COLS, colors):
    fig_trend.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df[metric],
        mode="lines+markers",
        name=metric.replace("_", " ").title(),
        line=dict(color=color, width=2),
        marker=dict(size=7),
        hovertemplate=(
            f"<b>{metric.replace('_', ' ').title()}</b><br>"
            "Score: %{y:.3f}<br>"
            "Run: %{text}<extra></extra>"
        ),
        text=df["run_label"],
    ))

fig_trend.update_layout(
    title="RAGAS Score Progression",
    xaxis_title="Timestamp",
    yaxis_title="Score (0 – 1)",
    yaxis_range=[0, 1.05],
    legend_title="Metric",
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_trend, use_container_width=True)

# --------------------------------------------------------
# Latency vs Quality Trade-off — Scatter
# --------------------------------------------------------

st.subheader("⚡ Latency vs. Quality Trade-off")

fig_scatter = px.scatter(
    df,
    x="avg_latency_ms",
    y="faithfulness",
    color="run_label",
    size="context_precision",
    hover_data=["retrieval_mode", "use_reranker", "top_k", "answer_relevancy"],
    title="Faithfulness vs. Latency per Configuration",
    size_max=25,
    color_discrete_sequence=px.colors.qualitative.Pastel,
)
fig_scatter.update_layout(
    xaxis_title="Average Latency (ms)",
    yaxis_title="Faithfulness Score",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_scatter, use_container_width=True)

# --------------------------------------------------------
# Comparison Table — Delta vs. Previous Run
# --------------------------------------------------------

st.subheader("🔍 Run Comparison Table")

if len(df) > 1:
    compare_df = df[["timestamp", "run_label", "retrieval_mode", "use_reranker", "top_k"] + METRIC_COLS + ["avg_latency_ms"]].copy()
    compare_df = compare_df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    # Compute deltas against the previous run
    for metric in METRIC_COLS:
        compare_df[f"Δ {metric}"] = compare_df[metric].diff(-1).round(3)

    # Style positive deltas green, negative red
    def color_delta(val):
        if pd.isna(val):
            return ""
        color = "#2ecc71" if val > 0 else ("#e74c3c" if val < 0 else "")
        return f"color: {color}; font-weight: bold"

    delta_cols = [c for c in compare_df.columns if c.startswith("Δ")]
    styled = compare_df.style.applymap(color_delta, subset=delta_cols).format(
        {m: "{:.3f}" for m in METRIC_COLS} | {"avg_latency_ms": "{:.0f}"}
    )
    st.dataframe(styled, use_container_width=True)
else:
    st.info("Run at least 2 eval configurations to see comparison deltas.")

st.divider()

# --------------------------------------------------------
# Full History Table
# --------------------------------------------------------

st.subheader("📋 Full Run History")
st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
