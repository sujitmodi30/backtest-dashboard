import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Backtest Dashboard", layout="wide")

DATA_DIR = Path(__file__).parent

STRATEGIES = [
    # (display_name, file, date_col, pnl_col, aggregate_legs_by_date, category)
    ("Nifty Momentum 6bar",              "nifty_momentum_5min_6bar.csv",                  "date",       "pnl",     False, "Momentum"),
    ("Nifty Momentum + Midpoint Entry",  "nifty_momentum_5min_6bar_midpoint_entry.csv",   "date",       "pnl",     False, "Momentum"),
    ("Nifty Momentum + Midpoint Exit",   "nifty_momentum_5min_6bar_midpoint_exit.csv",    "date",       "pnl",     False, "Momentum"),
    ("Nifty Momentum + Prev High Exit",  "nifty_momentum_5min_6bar_prev_high_exit.csv",   "date",       "pnl",     False, "Momentum"),
    ("Nifty Momentum + Trailing SL",     "nifty_momentum_5min_6bar_trailing_sl.csv",      "date",       "pnl",     False, "Momentum"),
    ("BNF Supertrend Cross",             "banknifty_supertrend_cross.csv",                "date",       "pnl",     False, "Supertrend"),
    ("Nifty ST Cross Long",              "nifty_st_cross_long.csv",                       "date",       "pnl",     False, "Supertrend"),
    ("Nifty ST Cross Short",             "nifty_st_cross_short.csv",                      "date",       "pnl",     False, "Supertrend"),
    ("Nifty ST Cross Combined",          "nifty_st_cross_combined.csv",                   "date",       "pnl",     False, "Supertrend"),
    ("Nifty ST Cross Inverse",           "nifty_st_cross_inverse.csv",                    "date",       "pnl",     False, "Supertrend"),
    ("Nifty Futures ST",                 "nifty_futures_st_trades.csv",                   "date",       "net_pnl", False, "Supertrend"),
    ("Nifty Box Sweep",                  "nifty_box_sweep_trades.csv",                    "date",       "net_pnl", False, "Box Arb"),
    ("BNF Box Sweep",                    "bnf_box_sweep_trades.csv",                      "date",       "net_pnl", False, "Box Arb"),
    ("MidCap Box Sweep",                 "midcpnifty_box_sweep_trades.csv",               "date",       "net_pnl", False, "Box Arb"),
    ("Sensex Box Sweep",                 "sensex_box_sweep_trades.csv",                   "date",       "net_pnl", False, "Box Arb"),
    ("Finnifty Box Sweep",               "finnifty_box_sweep_trades.csv",                 "date",       "net_pnl", False, "Box Arb"),
    ("BNF Box Sweep Depth",              "bnf_box_sweep_depth_trades.csv",                "date",       "net_pnl", False, "Box Arb"),
    ("MidCap Box Sweep Depth",           "midcpnifty_box_sweep_depth_trades.csv",         "date",       "net_pnl", False, "Box Arb"),
    ("Box Arb Tick",                     "box_arb_backtest_trades.csv",                   "date",       "gross_pnl", False, "Box Arb"),
    ("BNF Box Arb",                      "bnf_box_arb.csv",                               "date",       "net_pnl", False, "Box Arb"),
    ("BNF PCP Arb",                      "bnf_pcp_arb.csv",                               "date",       "net_pnl", False, "PCP Arb"),
    ("BNF PCP 2-leg",                    "bnf_pcp_2leg.csv",                              "date",       "net_pnl", False, "PCP Arb"),
    ("BNF PCP Synth Fut",                "bnf_pcp_synth_fut.csv",                         "date",       "net_pnl", False, "PCP Arb"),
    ("BNF Butterfly Arb",                "bnf_butterfly_arb.csv",                         "date",       "net_pnl", False, "Butterfly Arb"),
    ("Nifty Short Straddle",             "nifty_short_straddle.csv",                      "date",       "pnl",     False, "Straddle"),
    ("BNF 930 Hedge",                    "bnf_930_hedge.csv",                             "date",       "pnl",     True,  "Carry"),
    ("BNF 930 Hedge Carry",              "bnf_930_hedge_carry.csv",                       "entry_date", "net_pnl", True,  "Carry"),
    ("BNF 930 Synth Carry",              "bnf_930_synth_carry.csv",                       "entry_date", "net_pnl", True,  "Carry"),
    ("BNF Futures + Put",                "banknifty_futures_put.csv",                     "entry_date", "pnl",     False, "Carry"),
    ("Nifty Highbreak Short",            "nifty_highbreak_short.csv",                     "date",       "pnl",     False, "Directional"),
    ("Nifty Long DPT",                   "nifty_long_dpt.csv",                            "date",       "pnl",     False, "Directional"),
    ("Nifty Short DPT",                  "nifty_short_dpt.csv",                           "date",       "pnl",     False, "Directional"),
    ("Box Arb Apr 2026",                 "box_arb_backtest_20260424.csv",                 "entry_ts",   "net_pnl", False, "Box Arb"),
]

CATEGORY_COLORS = {
    "Momentum":      "#636EFA",
    "Supertrend":    "#EF553B",
    "Box Arb":       "#00CC96",
    "PCP Arb":       "#AB63FA",
    "Butterfly Arb": "#FFA15A",
    "Straddle":      "#19D3F3",
    "Carry":         "#FF6692",
    "Directional":   "#B6E880",
}


@st.cache_data
def load_strategy(file, date_col, pnl_col, group_by_date):
    path = DATA_DIR / file
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
        date_col = date_col.lower()
        pnl_col = pnl_col.lower()
        if date_col not in df.columns or pnl_col not in df.columns:
            return None
        raw = df[date_col].astype(str).str.strip()
        # Handle YYYYMMDD integer format
        if raw.str.match(r"^\d{8}$").all():
            df["_date"] = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
        else:
            df["_date"] = pd.to_datetime(raw, errors="coerce").dt.normalize()
        df["_pnl"] = pd.to_numeric(df[pnl_col], errors="coerce")
        df = df.dropna(subset=["_date", "_pnl"])
        if group_by_date:
            df = df.groupby("_date")["_pnl"].sum().reset_index()
            df.columns = ["date", "pnl"]
        else:
            df = df[["_date", "_pnl"]].copy()
            df.columns = ["date", "pnl"]
        df = df.sort_values("date").reset_index(drop=True)
        df["cumulative"] = df["pnl"].cumsum()
        return df
    except Exception:
        return None


def compute_metrics(df, name):
    if df is None or df.empty:
        return None
    total_pnl = df["pnl"].sum()
    n_trades = len(df)
    wins = (df["pnl"] > 0).sum()
    win_rate = wins / n_trades * 100 if n_trades else 0
    avg_pnl = df["pnl"].mean()
    max_dd = (df["cumulative"] - df["cumulative"].cummax()).min()
    date_start = df["date"].min().strftime("%Y-%m-%d")
    date_end = df["date"].max().strftime("%Y-%m-%d")
    return {
        "Strategy": name,
        "Trades": n_trades,
        "Net PnL (₹)": round(total_pnl, 0),
        "Win Rate (%)": round(win_rate, 1),
        "Avg PnL/Trade (₹)": round(avg_pnl, 0),
        "Max DD (₹)": round(max_dd, 0),
        "From": date_start,
        "To": date_end,
    }


# ── Load all data ──────────────────────────────────────────────────────────────
data = {}
for name, file, date_col, pnl_col, agg, category in STRATEGIES:
    df = load_strategy(file, date_col, pnl_col, agg)
    if df is not None and not df.empty:
        data[name] = {"df": df, "category": category}

# ── Build summary table ────────────────────────────────────────────────────────
rows = []
for name, file, date_col, pnl_col, agg, category in STRATEGIES:
    if name in data:
        m = compute_metrics(data[name]["df"], name)
        if m:
            m["Category"] = category
            rows.append(m)

summary_df = pd.DataFrame(rows)

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.title("Filters")
categories = sorted(summary_df["Category"].unique())
selected_cats = st.sidebar.multiselect("Category", categories, default=categories)
filtered_summary = summary_df[summary_df["Category"].isin(selected_cats)]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Backtest Dashboard")
st.caption(f"{len(filtered_summary)} strategies loaded · all PnL in ₹")

# ── KPI row ────────────────────────────────────────────────────────────────────
total_strategies = len(filtered_summary)
total_pnl = filtered_summary["Net PnL (₹)"].sum()
total_trades = filtered_summary["Trades"].sum()
best_strat = filtered_summary.loc[filtered_summary["Net PnL (₹)"].idxmax(), "Strategy"] if not filtered_summary.empty else "—"
worst_strat = filtered_summary.loc[filtered_summary["Net PnL (₹)"].idxmin(), "Strategy"] if not filtered_summary.empty else "—"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Strategies", total_strategies)
k2.metric("Total Net PnL", f"₹{total_pnl:,.0f}")
k3.metric("Total Trades", f"{total_trades:,}")
k4.metric("Best Strategy", best_strat)
k5.metric("Worst Strategy", worst_strat)

st.divider()

# ── Tab layout ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Summary Table", "PnL Comparison", "Cumulative PnL", "Monthly Breakdown"])

# ────────── TAB 1: Summary Table ──────────────────────────────────────────────
with tab1:
    display_cols = ["Category", "Strategy", "Trades", "Net PnL (₹)", "Win Rate (%)",
                    "Avg PnL/Trade (₹)", "Max DD (₹)", "From", "To"]
    styled = filtered_summary[display_cols].sort_values("Net PnL (₹)", ascending=False).reset_index(drop=True)

    def color_pnl(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return "color: #00cc96; font-weight: bold"
            elif val < 0:
                return "color: #ef553b; font-weight: bold"
        return ""

    st.dataframe(
        styled.style
            .applymap(color_pnl, subset=["Net PnL (₹)", "Avg PnL/Trade (₹)", "Max DD (₹)"])
            .format({"Net PnL (₹)": "₹{:,.0f}", "Avg PnL/Trade (₹)": "₹{:,.0f}",
                     "Max DD (₹)": "₹{:,.0f}", "Win Rate (%)": "{:.1f}%"}),
        use_container_width=True,
        height=600,
    )

# ────────── TAB 2: PnL Comparison ─────────────────────────────────────────────
with tab2:
    col_bar, col_win = st.columns(2)

    with col_bar:
        bar_df = filtered_summary[["Strategy", "Net PnL (₹)", "Category"]].sort_values("Net PnL (₹)")
        fig_bar = go.Figure()
        for cat in bar_df["Category"].unique():
            sub = bar_df[bar_df["Category"] == cat]
            fig_bar.add_trace(go.Bar(
                y=sub["Strategy"],
                x=sub["Net PnL (₹)"],
                name=cat,
                orientation="h",
                marker_color=CATEGORY_COLORS.get(cat, "#888"),
            ))
        fig_bar.update_layout(
            title="Net PnL by Strategy",
            xaxis_title="Net PnL (₹)",
            height=max(400, len(bar_df) * 22),
            legend_title="Category",
            barmode="relative",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_win:
        scatter_df = filtered_summary[["Strategy", "Net PnL (₹)", "Win Rate (%)", "Trades", "Category"]].dropna()
        fig_sc = px.scatter(
            scatter_df,
            x="Win Rate (%)",
            y="Net PnL (₹)",
            size="Trades",
            color="Category",
            hover_name="Strategy",
            color_discrete_map=CATEGORY_COLORS,
            title="Win Rate vs Net PnL",
        )
        fig_sc.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_sc.add_vline(x=50, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_sc, use_container_width=True)

    # Category breakdown
    cat_df = filtered_summary.groupby("Category").agg(
        Total_PnL=("Net PnL (₹)", "sum"),
        Strategies=("Strategy", "count"),
        Trades=("Trades", "sum"),
    ).reset_index().sort_values("Total_PnL", ascending=False)

    col_pie, col_cbar = st.columns(2)
    with col_pie:
        fig_pie = px.pie(
            cat_df[cat_df["Total_PnL"] > 0],
            names="Category",
            values="Total_PnL",
            title="PnL Share (profitable categories)",
            color="Category",
            color_discrete_map=CATEGORY_COLORS,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_cbar:
        fig_cbar = go.Figure(go.Bar(
            x=cat_df["Category"],
            y=cat_df["Total_PnL"],
            marker_color=[CATEGORY_COLORS.get(c, "#888") for c in cat_df["Category"]],
        ))
        fig_cbar.update_layout(title="Total PnL by Category", yaxis_title="₹")
        st.plotly_chart(fig_cbar, use_container_width=True)

# ────────── TAB 3: Cumulative PnL ─────────────────────────────────────────────
with tab3:
    st.subheader("Cumulative PnL curves")

    selected_names = [
        n for n in filtered_summary["Strategy"].tolist() if n in data
    ]
    chosen = st.multiselect(
        "Select strategies to plot",
        selected_names,
        default=selected_names[:10] if len(selected_names) > 10 else selected_names,
    )

    if chosen:
        fig_cum = go.Figure()
        for name in chosen:
            df = data[name]["df"]
            cat = data[name]["category"]
            fig_cum.add_trace(go.Scatter(
                x=df["date"],
                y=df["cumulative"],
                mode="lines",
                name=name,
                line=dict(color=CATEGORY_COLORS.get(cat, "#888")),
            ))
        fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_cum.update_layout(
            title="Cumulative Net PnL",
            xaxis_title="Date",
            yaxis_title="Cumulative PnL (₹)",
            height=550,
            legend=dict(orientation="v", x=1.01),
            hovermode="x unified",
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # Drawdown chart
        fig_dd = go.Figure()
        for name in chosen:
            df = data[name]["df"]
            cat = data[name]["category"]
            rolling_max = df["cumulative"].cummax()
            dd = df["cumulative"] - rolling_max
            fig_dd.add_trace(go.Scatter(
                x=df["date"],
                y=dd,
                mode="lines",
                name=name,
                fill="tozeroy",
                line=dict(color=CATEGORY_COLORS.get(cat, "#888")),
            ))
        fig_dd.update_layout(
            title="Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (₹)",
            height=350,
            hovermode="x unified",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

# ────────── TAB 4: Monthly Breakdown ─────────────────────────────────────────
with tab4:
    st.subheader("Monthly PnL Heatmap")

    monthly_rows = []
    for name, info in data.items():
        if name not in filtered_summary["Strategy"].values:
            continue
        df = info["df"].copy()
        df["month"] = df["date"].dt.to_period("M").astype(str)
        m = df.groupby("month")["pnl"].sum().reset_index()
        m["Strategy"] = name
        monthly_rows.append(m)

    if monthly_rows:
        monthly_df = pd.concat(monthly_rows, ignore_index=True)
        pivot = monthly_df.pivot_table(index="Strategy", columns="month", values="pnl", aggfunc="sum")
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[
                [0.0, "#ef553b"],
                [0.5, "#1a1a2e"],
                [1.0, "#00cc96"],
            ],
            zmid=0,
            hovertemplate="Strategy: %{y}<br>Month: %{x}<br>PnL: ₹%{z:,.0f}<extra></extra>",
            colorbar=dict(title="PnL (₹)"),
        ))
        fig_heat.update_layout(
            title="Monthly PnL by Strategy",
            xaxis_title="Month",
            height=max(400, len(pivot) * 22 + 150),
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Monthly totals bar
        monthly_total = monthly_df.groupby("month")["pnl"].sum().reset_index()
        monthly_total = monthly_total.sort_values("month")
        fig_mbar = go.Figure(go.Bar(
            x=monthly_total["month"],
            y=monthly_total["pnl"],
            marker_color=["#00cc96" if v > 0 else "#ef553b" for v in monthly_total["pnl"]],
        ))
        fig_mbar.update_layout(
            title="Total PnL Across All Strategies by Month",
            xaxis_title="Month",
            yaxis_title="₹",
            height=300,
        )
        st.plotly_chart(fig_mbar, use_container_width=True)
