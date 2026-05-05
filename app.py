import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time
import os
import google.genai as genai
from google.genai import types as gtypes
import boto3

st.set_page_config(page_title="Quant Research Platform", layout="wide", page_icon="📈")

DATA_DIR = Path(__file__).parent
ATHENA_DATABASE = "dhan_tick_data"
ATHENA_RESULTS  = "s3://dhan-tick-data-parquet/athena-results/"
AWS_REGION      = "ap-south-1"

# ─── AWS / Anthropic clients ──────────────────────────────────────────────────

def get_gemini_key():
    try:
        k = st.secrets.get("GEMINI_API_KEY", "")
        if k:
            return k
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")

def get_aws_session():
    try:
        key    = st.secrets.get("AWS_ACCESS_KEY_ID", "")
        secret = st.secrets.get("AWS_SECRET_ACCESS_KEY", "")
        if key and secret:
            return boto3.Session(
                aws_access_key_id=key,
                aws_secret_access_key=secret,
                region_name=AWS_REGION,
            )
    except Exception:
        pass
    return boto3.Session(profile_name="default", region_name=AWS_REGION)

@st.cache_resource
def get_athena_client():
    return get_aws_session().client("athena")


def run_athena_query(sql: str, timeout: int = 120) -> pd.DataFrame:
    athena = get_athena_client()
    resp = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DATABASE},
        ResultConfiguration={"OutputLocation": ATHENA_RESULTS},
    )
    qid = resp["QueryExecutionId"]
    deadline = time.time() + timeout
    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        state  = status["State"]
        if state == "SUCCEEDED":
            break
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Athena query {state}: {status.get('StateChangeReason','')}")
        if time.time() > deadline:
            raise TimeoutError(f"Athena query timed out after {timeout}s")
        time.sleep(2)

    rows, header = [], None
    paginator = athena.get_paginator("get_query_results")
    for page in paginator.paginate(QueryExecutionId=qid):
        for row in page["ResultSet"]["Rows"]:
            vals = [c.get("VarCharValue", "") for c in row["Data"]]
            if header is None:
                header = vals
            else:
                rows.append(vals)
    return pd.DataFrame(rows, columns=header) if header else pd.DataFrame()

# ─── Dashboard ────────────────────────────────────────────────────────────────

STRATEGIES = [
    ("Nifty Momentum 6bar",             "nifty_momentum_5min_6bar.csv",                "date",       "pnl",      False, "Momentum"),
    ("Nifty Momentum + Midpoint Entry", "nifty_momentum_5min_6bar_midpoint_entry.csv", "date",       "pnl",      False, "Momentum"),
    ("Nifty Momentum + Midpoint Exit",  "nifty_momentum_5min_6bar_midpoint_exit.csv",  "date",       "pnl",      False, "Momentum"),
    ("Nifty Momentum + Prev High Exit", "nifty_momentum_5min_6bar_prev_high_exit.csv", "date",       "pnl",      False, "Momentum"),
    ("Nifty Momentum + Trailing SL",    "nifty_momentum_5min_6bar_trailing_sl.csv",    "date",       "pnl",      False, "Momentum"),
    ("BNF Supertrend Cross",            "banknifty_supertrend_cross.csv",              "date",       "pnl",      False, "Supertrend"),
    ("Nifty ST Cross Long",             "nifty_st_cross_long.csv",                     "date",       "pnl",      False, "Supertrend"),
    ("Nifty ST Cross Short",            "nifty_st_cross_short.csv",                    "date",       "pnl",      False, "Supertrend"),
    ("Nifty ST Cross Combined",         "nifty_st_cross_combined.csv",                 "date",       "pnl",      False, "Supertrend"),
    ("Nifty ST Cross Inverse",          "nifty_st_cross_inverse.csv",                  "date",       "pnl",      False, "Supertrend"),
    ("Nifty Futures ST",                "nifty_futures_st_trades.csv",                 "date",       "net_pnl",  False, "Supertrend"),
    ("Nifty Box Sweep",                 "nifty_box_sweep_trades.csv",                  "date",       "net_pnl",  False, "Box Arb"),
    ("BNF Box Sweep",                   "bnf_box_sweep_trades.csv",                    "date",       "net_pnl",  False, "Box Arb"),
    ("MidCap Box Sweep",                "midcpnifty_box_sweep_trades.csv",             "date",       "net_pnl",  False, "Box Arb"),
    ("Sensex Box Sweep",                "sensex_box_sweep_trades.csv",                 "date",       "net_pnl",  False, "Box Arb"),
    ("Finnifty Box Sweep",              "finnifty_box_sweep_trades.csv",               "date",       "net_pnl",  False, "Box Arb"),
    ("BNF Box Sweep Depth",             "bnf_box_sweep_depth_trades.csv",              "date",       "net_pnl",  False, "Box Arb"),
    ("MidCap Box Sweep Depth",          "midcpnifty_box_sweep_depth_trades.csv",       "date",       "net_pnl",  False, "Box Arb"),
    ("Box Arb Tick",                    "box_arb_backtest_trades.csv",                 "date",       "gross_pnl",False, "Box Arb"),
    ("BNF Box Arb",                     "bnf_box_arb.csv",                             "date",       "net_pnl",  False, "Box Arb"),
    ("Box Arb Apr 2026",                "box_arb_backtest_20260424.csv",               "entry_ts",   "net_pnl",  False, "Box Arb"),
    ("BNF PCP Arb",                     "bnf_pcp_arb.csv",                             "date",       "net_pnl",  False, "PCP Arb"),
    ("BNF PCP 2-leg",                   "bnf_pcp_2leg.csv",                            "date",       "net_pnl",  False, "PCP Arb"),
    ("BNF PCP Synth Fut",               "bnf_pcp_synth_fut.csv",                       "date",       "net_pnl",  False, "PCP Arb"),
    ("BNF Butterfly Arb",               "bnf_butterfly_arb.csv",                       "date",       "net_pnl",  False, "Butterfly Arb"),
    ("Nifty Short Straddle",            "nifty_short_straddle.csv",                    "date",       "pnl",      False, "Straddle"),
    ("BNF 930 Hedge",                   "bnf_930_hedge.csv",                           "date",       "pnl",      True,  "Carry"),
    ("BNF 930 Hedge Carry",             "bnf_930_hedge_carry.csv",                     "entry_date", "net_pnl",  True,  "Carry"),
    ("BNF 930 Synth Carry",             "bnf_930_synth_carry.csv",                     "entry_date", "net_pnl",  True,  "Carry"),
    ("BNF Futures + Put",               "banknifty_futures_put.csv",                   "entry_date", "pnl",      False, "Carry"),
    ("Nifty Highbreak Short",           "nifty_highbreak_short.csv",                   "date",       "pnl",      False, "Directional"),
    ("Nifty Long DPT",                  "nifty_long_dpt.csv",                          "date",       "pnl",      False, "Directional"),
    ("Nifty Short DPT",                 "nifty_short_dpt.csv",                         "date",       "pnl",      False, "Directional"),
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
        date_col, pnl_col = date_col.lower(), pnl_col.lower()
        if date_col not in df.columns or pnl_col not in df.columns:
            return None
        raw = df[date_col].astype(str).str.strip()
        df["_date"] = (
            pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
            if raw.str.match(r"^\d{8}$").all()
            else pd.to_datetime(raw, errors="coerce").dt.normalize()
        )
        df["_pnl"] = pd.to_numeric(df[pnl_col], errors="coerce")
        df = df.dropna(subset=["_date", "_pnl"])
        if group_by_date:
            df = df.groupby("_date")["_pnl"].sum().reset_index()
            df.columns = ["date", "pnl"]
        else:
            df = df[["_date", "_pnl"]].rename(columns={"_date": "date", "_pnl": "pnl"})
        df = df.sort_values("date").reset_index(drop=True)
        df["cumulative"] = df["pnl"].cumsum()
        return df
    except Exception:
        return None

def compute_metrics(df, name, category):
    if df is None or df.empty:
        return None
    n   = len(df)
    win = (df["pnl"] > 0).sum()
    cum = df["cumulative"]
    return {
        "Strategy":          name,
        "Category":          category,
        "Trades":            n,
        "Net PnL (₹)":       round(df["pnl"].sum(), 0),
        "Win Rate (%)":      round(win / n * 100, 1) if n else 0,
        "Avg PnL/Trade (₹)": round(df["pnl"].mean(), 0),
        "Max DD (₹)":        round((cum - cum.cummax()).min(), 0),
        "From":              df["date"].min().strftime("%Y-%m-%d"),
        "To":                df["date"].max().strftime("%Y-%m-%d"),
    }

def show_dashboard():
    st.title("📊 Backtest Dashboard")

    data = {}
    for name, file, dc, pc, agg, cat in STRATEGIES:
        df = load_strategy(file, dc, pc, agg)
        if df is not None and not df.empty:
            data[name] = {"df": df, "category": cat}

    rows = [compute_metrics(v["df"], k, v["category"]) for k, v in data.items()]
    summary = pd.DataFrame([r for r in rows if r])

    # Sidebar filters
    cats = sorted(summary["Category"].unique())
    sel  = st.sidebar.multiselect("Category", cats, default=cats)
    filt = summary[summary["Category"].isin(sel)]

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Strategies",    len(filt))
    c2.metric("Total Net PnL", f"₹{filt['Net PnL (₹)'].sum():,.0f}")
    c3.metric("Total Trades",  f"{filt['Trades'].sum():,}")
    if not filt.empty:
        c4.metric("Best",  filt.loc[filt["Net PnL (₹)"].idxmax(), "Strategy"])
        c5.metric("Worst", filt.loc[filt["Net PnL (₹)"].idxmin(), "Strategy"])

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["Summary Table", "PnL Comparison", "Cumulative PnL", "Monthly Breakdown"])

    with tab1:
        cols = ["Category","Strategy","Trades","Net PnL (₹)","Win Rate (%)","Avg PnL/Trade (₹)","Max DD (₹)","From","To"]
        styled = filt[cols].sort_values("Net PnL (₹)", ascending=False).reset_index(drop=True)
        def color_pnl(v):
            if isinstance(v, (int,float)):
                return "color:#00cc96;font-weight:bold" if v>0 else ("color:#ef553b;font-weight:bold" if v<0 else "")
            return ""
        st.dataframe(
            styled.style
                .applymap(color_pnl, subset=["Net PnL (₹)","Avg PnL/Trade (₹)","Max DD (₹)"])
                .format({"Net PnL (₹)":"₹{:,.0f}","Avg PnL/Trade (₹)":"₹{:,.0f}","Max DD (₹)":"₹{:,.0f}","Win Rate (%)":"{:.1f}%"}),
            use_container_width=True, height=600,
        )

    with tab2:
        c_bar, c_sc = st.columns(2)
        with c_bar:
            bar = filt[["Strategy","Net PnL (₹)","Category"]].sort_values("Net PnL (₹)")
            fig = go.Figure()
            for cat in bar["Category"].unique():
                sub = bar[bar["Category"]==cat]
                fig.add_trace(go.Bar(y=sub["Strategy"], x=sub["Net PnL (₹)"], name=cat,
                                     orientation="h", marker_color=CATEGORY_COLORS.get(cat,"#888")))
            fig.update_layout(title="Net PnL by Strategy", height=max(400,len(bar)*22))
            st.plotly_chart(fig, use_container_width=True)
        with c_sc:
            fig2 = px.scatter(filt, x="Win Rate (%)", y="Net PnL (₹)", size="Trades",
                              color="Category", hover_name="Strategy",
                              color_discrete_map=CATEGORY_COLORS, title="Win Rate vs Net PnL")
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            fig2.add_vline(x=50, line_dash="dash", line_color="gray")
            st.plotly_chart(fig2, use_container_width=True)

        cat_agg = filt.groupby("Category").agg(Total_PnL=("Net PnL (₹)","sum"), Trades=("Trades","sum")).reset_index()
        c_pie, c_cbar = st.columns(2)
        with c_pie:
            pos = cat_agg[cat_agg["Total_PnL"]>0]
            fig3 = px.pie(pos, names="Category", values="Total_PnL", color="Category",
                          color_discrete_map=CATEGORY_COLORS, title="PnL Share (profitable)")
            st.plotly_chart(fig3, use_container_width=True)
        with c_cbar:
            fig4 = go.Figure(go.Bar(x=cat_agg["Category"], y=cat_agg["Total_PnL"],
                                    marker_color=[CATEGORY_COLORS.get(c,"#888") for c in cat_agg["Category"]]))
            fig4.update_layout(title="Total PnL by Category", yaxis_title="₹")
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        names = [n for n in filt["Strategy"].tolist() if n in data]
        chosen = st.multiselect("Strategies", names, default=names[:10] if len(names)>10 else names)
        if chosen:
            fig5 = go.Figure()
            for n in chosen:
                df = data[n]["df"]
                cat = data[n]["category"]
                fig5.add_trace(go.Scatter(x=df["date"], y=df["cumulative"], name=n, mode="lines",
                                          line=dict(color=CATEGORY_COLORS.get(cat,"#888"))))
            fig5.add_hline(y=0, line_dash="dash", line_color="gray")
            fig5.update_layout(title="Cumulative PnL", height=500, hovermode="x unified")
            st.plotly_chart(fig5, use_container_width=True)

            fig6 = go.Figure()
            for n in chosen:
                df = data[n]["df"]
                dd = df["cumulative"] - df["cumulative"].cummax()
                fig6.add_trace(go.Scatter(x=df["date"], y=dd, name=n, fill="tozeroy",
                                          line=dict(color=CATEGORY_COLORS.get(data[n]["category"],"#888"))))
            fig6.update_layout(title="Drawdown", height=300, hovermode="x unified")
            st.plotly_chart(fig6, use_container_width=True)

    with tab4:
        monthly = []
        for n, info in data.items():
            if n not in filt["Strategy"].values:
                continue
            df = info["df"].copy()
            df["month"] = df["date"].dt.to_period("M").astype(str)
            m = df.groupby("month")["pnl"].sum().reset_index()
            m["Strategy"] = n
            monthly.append(m)
        if monthly:
            mdf = pd.concat(monthly, ignore_index=True)
            pivot = mdf.pivot_table(index="Strategy", columns="month", values="pnl", aggfunc="sum")
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)
            fig7 = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                colorscale=[[0,"#ef553b"],[0.5,"#1a1a2e"],[1,"#00cc96"]], zmid=0,
                hovertemplate="Strategy: %{y}<br>Month: %{x}<br>PnL: ₹%{z:,.0f}<extra></extra>",
            ))
            fig7.update_layout(title="Monthly PnL Heatmap", height=max(400, len(pivot)*22+150))
            st.plotly_chart(fig7, use_container_width=True)

            tot = mdf.groupby("month")["pnl"].sum().reset_index().sort_values("month")
            fig8 = go.Figure(go.Bar(
                x=tot["month"], y=tot["pnl"],
                marker_color=["#00cc96" if v>0 else "#ef553b" for v in tot["pnl"]],
            ))
            fig8.update_layout(title="Total Monthly PnL (all strategies)", height=280)
            st.plotly_chart(fig8, use_container_width=True)

# ─── AI Query ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert quant analyst AI with access to an Indian options trading data lake on AWS Athena.

## Database: dhan_tick_data

### dhan_option_ticks_index  — LTP tick data for index options
Partitioned by trade_date (VARCHAR 'YYYYMMDD', e.g. '20251110'). Always filter this.
Date range: 20251110 → present.
Symbols: NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY, SENSEX, BANKEX.

| Column        | Type    | Notes |
|---------------|---------|-------|
| ts            | TIMESTAMP WITH TIME ZONE | tick time (IST) |
| symbol        | VARCHAR | NIFTY, BANKNIFTY etc. |
| option_symbol | VARCHAR | e.g. 'BANKNIFTY 25 NOV 57900 CALL' |
| strike_price  | BIGINT  | |
| option_type   | VARCHAR | CE or PE |
| ltp           | DOUBLE  | last traded price — USE THIS for candles |
| ltq           | BIGINT  | last traded qty |
| volume        | BIGINT  | cumulative day volume |
| oi            | BIGINT  | open interest |
| atm_offset    | INTEGER | 0=ATM, negative=below ATM, positive=above ATM |
| expiry_date   | VARCHAR | |

**Note:** open/high/low/close columns = day-level exchange stats. Do NOT use them for candles.

### dhan_option_depth_index  — Bid/ask depth (5 levels)
Partitioned by trade_date. Join on tick_id or security_id to ticks table.
Key cols: ts, symbol, option_symbol, option_type, level (1=best), bid_price, ask_price, bid_quantity, ask_quantity, atm_offset

## Athena / Presto SQL rules
1. Always filter: WHERE trade_date = '20251110'
2. IST timestamp: CAST(at_timezone(ts, 'Asia/Kolkata') AS TIMESTAMP)
3. 5-min candle bucket:
   date_trunc('minute', CAST(at_timezone(ts,'Asia/Kolkata') AS TIMESTAMP))
   - INTERVAL '1' MINUTE * (minute(CAST(at_timezone(ts,'Asia/Kolkata') AS TIMESTAMP)) % 5)
   AS candle_ts
4. Candle OHLC: min_by(ltp,ts) open, max_by(ltp,ts) close, MIN(ltp) low, MAX(ltp) high
5. Market hours 09:15–15:30 IST; last 5-min candle at 15:25
6. Add LIMIT (e.g. LIMIT 500) to avoid huge result sets
7. String dates in WHERE must use single quotes: trade_date = '20260115'

## Guidelines
- Explain what each query does before running it
- After results, give clear analysis and insights
- Suggest a chart type that best visualises the data
- If a query fails, diagnose the error and retry with a fix
"""

GEMINI_TOOL = gtypes.Tool(functionDeclarations=[
    gtypes.FunctionDeclaration(
        name="run_query",
        description="Execute a Presto SQL query against the dhan_tick_data Athena database and return the results.",
        parameters=gtypes.Schema(
            type="OBJECT",
            properties={
                "sql":         gtypes.Schema(type="STRING", description="The Presto SQL query to run"),
                "description": gtypes.Schema(type="STRING", description="One-line description of what this query does"),
            },
            required=["sql", "description"],
        ),
    )
])

def auto_chart(df: pd.DataFrame):
    if df is None or df.empty or len(df.columns) < 2:
        return None

    # Coerce numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    numeric = df.select_dtypes(include="number").columns.tolist()
    time_col = next((c for c in df.columns if any(t in c.lower() for t in ["ts","time","date","candle","bar"])), None)

    # Candlestick?
    ohlc = {"open","high","low","close"}
    if ohlc.issubset({c.lower() for c in df.columns}) and time_col:
        cm = {c.lower(): c for c in df.columns}
        return go.Figure(go.Candlestick(
            x=df[time_col], open=df[cm["open"]], high=df[cm["high"]],
            low=df[cm["low"]], close=df[cm["close"]],
        )).update_layout(title="Candlestick", xaxis_rangeslider_visible=False)

    # Line chart (time + numerics)
    if time_col and numeric:
        fig = go.Figure()
        for col in numeric[:4]:
            fig.add_trace(go.Scatter(x=df[time_col], y=df[col], name=col, mode="lines+markers"))
        return fig.update_layout(title="Time Series", hovermode="x unified")

    # Bar chart (categorical + numeric)
    cat = next((c for c in df.columns if c not in numeric), None)
    if cat and numeric:
        return px.bar(df, x=cat, y=numeric[0], title=f"{numeric[0]} by {cat}")

    return None

def run_ai_loop(user_message: str, chat_history: list):
    """Run full Gemini agentic loop. Yields (type, payload) events for streaming display."""
    key = get_gemini_key()
    if not key:
        yield ("error", "GEMINI_API_KEY not set. Add it to Streamlit secrets.")
        return

    client = genai.Client(api_key=key)
    last_df = None

    # Reconstruct chat from history and send current message
    contents = list(chat_history) + [gtypes.Content(
        role="user",
        parts=[gtypes.Part(text=user_message)],
    )]

    while True:
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=contents,
                config=gtypes.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=[GEMINI_TOOL],
                    temperature=0.1,
                ),
            )
        except Exception as e:
            yield ("error", f"Gemini API error: {type(e).__name__}: {e}")
            return

        candidate = response.candidates[0]
        contents.append(candidate.content)  # add assistant turn to history

        # Check for function calls
        fn_parts = [p for p in candidate.content.parts if p.function_call is not None]

        if fn_parts:
            tool_response_parts = []
            for part in fn_parts:
                fc   = part.function_call
                sql  = fc.args.get("sql", "")
                desc = fc.args.get("description", "Running query…")
                yield ("sql", (desc, sql))

                try:
                    df      = run_athena_query(sql)
                    last_df = df
                    yield ("result", df)
                    result_text = f"Query returned {len(df)} rows:\n{df.head(300).to_csv(index=False)}"
                except Exception as e:
                    yield ("error", str(e))
                    result_text = f"Error executing query: {e}"

                tool_response_parts.append(gtypes.Part(
                    function_response=gtypes.FunctionResponse(
                        name=fc.name,
                        response={"result": result_text},
                    )
                ))

            # Send tool results back
            contents.append(gtypes.Content(role="user", parts=tool_response_parts))

        else:
            # Final text response
            text = "".join(p.text for p in candidate.content.parts if hasattr(p, "text") and p.text)
            fig  = auto_chart(last_df)
            yield ("final", (text, last_df, fig, contents))
            return

def show_ai_query():
    st.title("🤖 AI Query")
    st.caption("Ask anything in plain English. Claude will query your S3 data lake and analyse the results.")

    with st.expander("💡 Example queries", expanded=False):
        st.markdown("""
- *Show 5-min candles for NIFTY ATM call on 2026-01-15*
- *What was total BANKNIFTY options volume on 2026-02-01?*
- *Compare ATM CE vs PE LTP for NIFTY on 2025-12-15*
- *Find the top 5 most liquid NIFTY strikes on 2026-03-01*
- *Show OI by strike for BANKNIFTY on 2026-01-20*
- *What was bid-ask spread on NIFTY ATM options at 9:30 on 2026-01-10?*
        """)

    if "chat" not in st.session_state:
        st.session_state.chat = []        # [(role, text, df, fig)]
    if "api_msgs" not in st.session_state:
        st.session_state.api_msgs = []    # full message history for API

    _, col_clear = st.columns([8, 1])
    with col_clear:
        if st.button("Clear"):
            st.session_state.chat     = []
            st.session_state.api_msgs = []
            st.rerun()

    # Render chat history
    for role, text, df, fig in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(text)
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True, height=300)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    # Input
    if prompt := st.chat_input("Ask about your options data…"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat.append(("user", prompt, None, None))

        with st.chat_message("assistant"):
            final_text, final_df, final_fig, updated_msgs = None, None, None, None

            for event_type, payload in run_ai_loop(prompt, st.session_state.api_msgs):
                if event_type == "sql":
                    desc, sql = payload
                    st.markdown(f"**Running:** {desc}")
                    st.code(sql, language="sql")

                elif event_type == "result":
                    df = payload
                    st.caption(f"↳ {len(df)} rows returned")
                    st.dataframe(df, use_container_width=True, height=200)
                    final_df = df

                elif event_type == "error":
                    st.error(payload)

                elif event_type == "final":
                    final_text, final_df, final_fig, updated_msgs = payload
                    st.markdown(final_text)
                    if final_df is not None and not final_df.empty:
                        st.dataframe(final_df, use_container_width=True, height=300)
                    if final_fig is not None:
                        st.plotly_chart(final_fig, use_container_width=True)

        if final_text:
            st.session_state.chat.append(("assistant", final_text, final_df, final_fig))
        if updated_msgs:
            st.session_state.api_msgs = updated_msgs  # full Gemini contents history

# ─── Navigation ───────────────────────────────────────────────────────────────

st.sidebar.title("📈 Quant Research")
page = st.sidebar.radio("", ["📊 Backtest Dashboard", "🤖 AI Query"], label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption("Data: S3 / Athena  ·  AI: Claude Sonnet 4.6")

if page == "📊 Backtest Dashboard":
    show_dashboard()
else:
    show_ai_query()
