"""
DOTr PMR CY 2021 — Interactive Visualization Dashboard
=======================================================
Launches a local Dash web app with six coordinated charts:

  Tab 1 — Overview
    • KPI cards (total contracts, total ABC, savings, failure rate)
    • Donut — status breakdown
    • Bar  — contracts by BAC unit (colored by status)

  Tab 2 — Budget & Savings
    • Horizontal bar — ABC vs Contract price per completed contract
    • Scatter — ABC vs savings % (bubble = contract size)

  Tab 3 — Risk & Failure
    • Heatmap — failure rate by BAC × procurement mode
    • Bar — failure reason frequency

  Tab 4 — Timeline
    • Gantt-style strip — pre-proc → NTP per completed contract

Install:
    pip install dash plotly pandas

Run:
    python dotr_pmr_2021_dashboard.py
Then open  http://127.0.0.1:8050  in your browser.
"""

import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    import dash
    from dash import dcc, html, Input, Output, callback
except ImportError:
    print("Dash not found.  Run:  pip install dash plotly pandas")
    sys.exit(1)

try:
    from dotr_pmr_2021_data import records, build_dataframe
except ModuleNotFoundError:
    print("[ERROR] dotr_pmr_2021_data.py not found. Place both files in the same folder.")
    sys.exit(1)


# ────────────────────────────────────────────────────────────
# 1.  DATA PREP
# ────────────────────────────────────────────────────────────

df = build_dataframe(records)
df = df[df["abc_php"].notna()].copy().reset_index(drop=True)

# Short BAC labels for readability
BAC_SHORT = {
    "BAC - Central Office"          : "BAC-CO",
    "CBAC - Aviation and Airports"  : "CBAC-Aviation",
    "CBAC - Railways"               : "CBAC-Rail",
    "SBAC - BRT"                    : "SBAC-BRT",
    "SBAC - BRT (Below 250M)"       : "SBAC-BRT<250M",
    "SBAC - NCICPP"                 : "SBAC-NCICPP",
    "SBAC - MRP-TDD"                : "SBAC-MRP",
    "SBAC - EDSA Greenways"         : "SBAC-EDSA",
}
df["bac_short"] = df["bac"].map(BAC_SHORT).fillna(df["bac"])

STATUS_COLORS = {
    "Completed"         : "#2ECC71",
    "On-Going"          : "#3498DB",
    "Failed"            : "#E74C3C",
    "Cancelled"         : "#E67E22",
    "Early Procurement" : "#9B59B6",
}

completed = df[df["status"] == "Completed"].copy()
failed    = df[df["status"] == "Failed"].copy()

# ────────────────────────────────────────────────────────────
# 2.  KPI VALUES
# ────────────────────────────────────────────────────────────

total_contracts   = len(df)
total_abc         = df["abc_php"].sum()
total_savings     = completed["savings_php"].sum()
failure_rate      = len(failed) / len(df) * 100
avg_savings_pct   = completed["savings_pct"].mean()
total_contract_v  = completed["contract_php"].sum()


# ────────────────────────────────────────────────────────────
# 3.  CHART BUILDERS
# ────────────────────────────────────────────────────────────

def make_kpi_cards():
    kpis = [
        ("Total Contracts",   f"{total_contracts}",                    "#1ABC9C"),
        ("Total ABC",         f"₱{total_abc/1e9:.2f}B",               "#3498DB"),
        ("Total Savings",     f"₱{total_savings/1e6:.1f}M",           "#2ECC71"),
        ("Failure Rate",      f"{failure_rate:.1f}%",                  "#E74C3C"),
        ("Avg Savings %",     f"{avg_savings_pct:.1f}%",               "#F39C12"),
        ("Completed Value",   f"₱{total_contract_v/1e9:.2f}B",        "#9B59B6"),
    ]
    cards = []
    for label, value, color in kpis:
        cards.append(
            html.Div([
                html.P(value, style={
                    "fontSize": "2rem", "fontWeight": "800",
                    "color": color, "margin": "0", "lineHeight": "1"
                }),
                html.P(label, style={
                    "fontSize": "0.75rem", "color": "#8899AA",
                    "margin": "4px 0 0 0", "letterSpacing": "0.08em",
                    "textTransform": "uppercase"
                }),
            ], style={
                "background": "#0F1923",
                "border": f"1px solid {color}30",
                "borderLeft": f"3px solid {color}",
                "borderRadius": "6px",
                "padding": "18px 20px",
                "flex": "1",
                "minWidth": "130px",
            })
        )
    return html.Div(cards, style={
        "display": "flex", "gap": "12px", "flexWrap": "wrap",
        "marginBottom": "24px"
    })


def fig_status_donut():
    counts = df["status"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.62,
        marker_colors=[STATUS_COLORS.get(s, "#95A5A6") for s in counts.index],
        textinfo="label+percent",
        textfont=dict(family="IBM Plex Mono", size=11, color="#ECEFF4"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**chart_layout("Status Breakdown"))
    fig.update_layout(
        annotations=[dict(text=f"<b>{total_contracts}</b><br><span style='font-size:10px'>contracts</span>",
                          x=0.5, y=0.5, font_size=18, showarrow=False,
                          font_color="#ECEFF4", font_family="IBM Plex Mono")]
    )
    return fig


def fig_bac_status_bar():
    pivot = (
        df.groupby(["bac_short", "status"])
        .size().reset_index(name="count")
    )
    fig = px.bar(
        pivot, x="count", y="bac_short", color="status",
        color_discrete_map=STATUS_COLORS,
        orientation="h",
        barmode="stack",
        labels={"count": "Contracts", "bac_short": "BAC Unit", "status": "Status"},
    )
    fig.update_layout(**chart_layout("Contracts by BAC Unit"))
    fig.update_traces(marker_line_width=0)
    return fig


def fig_abc_vs_contract():
    c = completed.copy()
    c["project_short"] = c["project"].str[:45] + "…"
    c = c.sort_values("abc_php", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=c["project_short"], x=c["abc_php"],
        name="ABC", orientation="h",
        marker_color="#3498DB44",
        marker_line_color="#3498DB", marker_line_width=1,
    ))
    fig.add_trace(go.Bar(
        y=c["project_short"], x=c["contract_php"],
        name="Contract Price", orientation="h",
        marker_color="#2ECC71",
    ))
    fig.update_layout(**chart_layout("ABC vs Contract Price — Completed Contracts"))
    fig.update_layout(
        barmode="overlay",
        xaxis_tickprefix="₱", xaxis_tickformat=",.0f",
        height=max(350, len(c) * 28),
    )
    return fig


def fig_savings_scatter():
    c = completed.copy()
    c["project_short"] = c["project"].str[:50]
    fig = px.scatter(
        c, x="abc_php", y="savings_pct",
        size="abc_php", color="bac_short",
        hover_name="project_short",
        hover_data={"abc_php": ":,.0f", "savings_pct": ":.2f", "bac_short": False},
        labels={"abc_php": "ABC (PHP)", "savings_pct": "Savings %", "bac_short": "BAC"},
        size_max=40,
    )
    fig.update_layout(**chart_layout("ABC vs Savings % — Completed Contracts"))
    fig.update_layout(xaxis_tickprefix="₱", xaxis_tickformat=",.0f")
    fig.update_traces(marker_line_width=0.5, marker_line_color="#ECEFF4")
    return fig


def fig_failure_heatmap():
    subset = df[df["status"].isin(["Failed", "Cancelled"])].copy()
    if subset.empty:
        return go.Figure()

    pivot = (
        subset.groupby(["bac_short", "mode"])
        .size().unstack(fill_value=0)
    )
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#0F1923"], [0.5, "#E67E22"], [1, "#E74C3C"]],
        text=pivot.values,
        texttemplate="%{text}",
        textfont=dict(color="#ECEFF4", size=11),
        hovertemplate="BAC: %{y}<br>Mode: %{x}<br>Failed/Cancelled: %{z}<extra></extra>",
    ))
    fig.update_layout(**chart_layout("Failure & Cancellation Heatmap — BAC × Mode"))
    fig.update_layout(
        xaxis_tickangle=-30,
        xaxis=dict(tickfont=dict(size=10)),
    )
    return fig


def fig_failure_reasons():
    f = failed[failed.get("failure_reason", pd.Series(dtype=str)).notna()] if "failure_reason" in failed.columns else pd.DataFrame()
    if f.empty or "failure_reason" not in f.columns:
        return go.Figure()
    counts = f["failure_reason"].value_counts()
    fig = px.bar(
        x=counts.values, y=counts.index,
        orientation="h",
        labels={"x": "Count", "y": "Failure Reason"},
        color=counts.values,
        color_continuous_scale=["#E67E22", "#E74C3C"],
    )
    fig.update_layout(**chart_layout("Failure Reasons (RA 9184)"))
    fig.update_coloraxes(showscale=False)
    fig.update_traces(marker_line_width=0)
    return fig


def fig_timeline():
    c = completed.copy()
    date_cols_needed = ["pre_proc_conference", "notice_to_proceed"]
    c = c.dropna(subset=date_cols_needed)
    if c.empty:
        return go.Figure()

    c = c.sort_values("pre_proc_conference")
    c["project_short"] = c["project"].str[:40] + "…"

    STAGES = [
        ("pre_proc_conference",    "posting_ib",               "Pre-Proc → IB",     "#3498DB"),
        ("posting_ib",             "pre_bid_conference",        "IB → Pre-Bid",      "#1ABC9C"),
        ("pre_bid_conference",     "submission_opening_bids",   "Pre-Bid → Open",    "#F39C12"),
        ("submission_opening_bids","bac_resolution",            "Open → Resolution", "#9B59B6"),
        ("bac_resolution",         "contract_signing",          "Resolution → Sign", "#E74C3C"),
        ("contract_signing",       "notice_to_proceed",         "Sign → NTP",        "#2ECC71"),
    ]

    fig = go.Figure()
    for row_i, (_, row) in enumerate(c.iterrows()):
        for start_col, end_col, stage_name, color in STAGES:
            s = row.get(start_col)
            e = row.get(end_col)
            if pd.notna(s) and pd.notna(e) and e >= s:
                fig.add_trace(go.Bar(
                    x=[(e - s).days],
                    base=[s.timestamp() * 1000],
                    y=[row["project_short"]],
                    orientation="h",
                    marker_color=color,
                    name=stage_name,
                    legendgroup=stage_name,
                    showlegend=(row_i == 0),
                    hovertemplate=(
                        f"<b>{row['project_short']}</b><br>"
                        f"Stage: {stage_name}<br>"
                        f"Start: {s.date()}<br>"
                        f"End: {e.date()}<br>"
                        f"Duration: {(e-s).days} days<extra></extra>"
                    ),
                ))

    fig.update_layout(**chart_layout("Procurement Timeline — Completed Contracts"))
    fig.update_layout(
        barmode="overlay",
        xaxis=dict(
            type="date",
            tickformat="%b %Y",
            tickfont=dict(size=10),
        ),
        height=max(400, len(c) * 30),
        legend=dict(orientation="h", y=-0.08, font_size=10),
    )
    return fig


# ────────────────────────────────────────────────────────────
# 4.  SHARED LAYOUT THEME
# ────────────────────────────────────────────────────────────

def chart_layout(title: str) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(family="IBM Plex Mono", size=13, color="#ECEFF4"),
            x=0,
        ),
        paper_bgcolor="#0A1520",
        plot_bgcolor="#0F1923",
        font=dict(family="IBM Plex Mono", color="#8899AA", size=11),
        margin=dict(l=20, r=20, t=48, b=20),
        xaxis=dict(gridcolor="#1C2D3D", zerolinecolor="#1C2D3D"),
        yaxis=dict(gridcolor="#1C2D3D", zerolinecolor="#1C2D3D"),
        legend=dict(bgcolor="#0A1520", bordercolor="#1C2D3D", borderwidth=1),
        height=380,
    )


# ────────────────────────────────────────────────────────────
# 5.  DASH APP LAYOUT
# ────────────────────────────────────────────────────────────

DARK_BG   = "#060E18"
PANEL_BG  = "#0A1520"
ACCENT    = "#00D4FF"
TEXT      = "#ECEFF4"
MONO      = "IBM Plex Mono, Courier New, monospace"

app = dash.Dash(__name__, title="DOTr PMR 2021 Dashboard")

tab_style = {
    "backgroundColor": PANEL_BG,
    "color": "#8899AA",
    "fontFamily": MONO,
    "fontSize": "0.8rem",
    "letterSpacing": "0.06em",
    "padding": "10px 18px",
    "border": "none",
    "borderBottom": f"2px solid transparent",
}
tab_selected_style = {
    **tab_style,
    "color": ACCENT,
    "borderBottom": f"2px solid {ACCENT}",
    "backgroundColor": PANEL_BG,
}

graph_config = {"displayModeBar": False}

app.layout = html.Div(style={
    "backgroundColor": DARK_BG,
    "minHeight": "100vh",
    "padding": "32px 40px",
    "fontFamily": MONO,
    "color": TEXT,
}, children=[

    # ── HEADER ──────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div(style={
                "width": "4px", "height": "48px",
                "backgroundColor": ACCENT, "marginRight": "16px",
                "borderRadius": "2px",
            }),
            html.Div([
                html.H1("DOTr PMR CY 2021", style={
                    "fontSize": "1.5rem", "fontWeight": "800",
                    "color": TEXT, "margin": "0", "letterSpacing": "0.04em"
                }),
                html.P("Procurement Monitoring Report · Interactive Dashboard",
                       style={"fontSize": "0.75rem", "color": "#8899AA", "margin": "2px 0 0 0"}),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Span("■ ", style={"color": STATUS_COLORS["Completed"]}),
            html.Span("Completed  ", style={"fontSize": "0.7rem", "marginRight": "12px"}),
            html.Span("■ ", style={"color": STATUS_COLORS["On-Going"]}),
            html.Span("On-Going  ", style={"fontSize": "0.7rem", "marginRight": "12px"}),
            html.Span("■ ", style={"color": STATUS_COLORS["Failed"]}),
            html.Span("Failed  ", style={"fontSize": "0.7rem", "marginRight": "12px"}),
            html.Span("■ ", style={"color": STATUS_COLORS["Cancelled"]}),
            html.Span("Cancelled  ", style={"fontSize": "0.7rem"}),
        ], style={"fontSize": "0.75rem", "color": TEXT}),
    ], style={"display": "flex", "justifyContent": "space-between",
              "alignItems": "center", "marginBottom": "28px"}),

    # ── KPI CARDS ───────────────────────────────────────────
    make_kpi_cards(),

    # ── FILTER BAR ──────────────────────────────────────────
    html.Div([
        html.Span("Filter by BAC:", style={"fontSize": "0.75rem", "color": "#8899AA",
                                            "alignSelf": "center", "marginRight": "10px"}),
        dcc.Dropdown(
            id="bac-filter",
            options=[{"label": "All BAC Units", "value": "ALL"}] +
                    [{"label": v, "value": v} for v in sorted(df["bac_short"].unique())],
            value="ALL",
            clearable=False,
            style={
                "width": "260px",
                "fontFamily": MONO,
                "fontSize": "0.78rem",
                "backgroundColor": PANEL_BG,
                "color": TEXT,
                "border": f"1px solid #1C2D3D",
            },
        ),
        html.Span("Semester:", style={"fontSize": "0.75rem", "color": "#8899AA",
                                       "alignSelf": "center", "marginLeft": "16px",
                                       "marginRight": "10px"}),
        dcc.Dropdown(
            id="sem-filter",
            options=[
                {"label": "Both Semesters", "value": "ALL"},
                {"label": "1st Semester",   "value": "1st"},
                {"label": "2nd Semester",   "value": "2nd"},
            ],
            value="ALL",
            clearable=False,
            style={
                "width": "200px",
                "fontFamily": MONO,
                "fontSize": "0.78rem",
                "backgroundColor": PANEL_BG,
                "color": TEXT,
                "border": f"1px solid #1C2D3D",
            },
        ),
    ], style={"display": "flex", "alignItems": "center",
              "marginBottom": "20px", "gap": "0"}),

    # ── TABS ────────────────────────────────────────────────
    dcc.Tabs(id="tabs", value="overview", children=[
        dcc.Tab(label="OVERVIEW",       value="overview",
                style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label="BUDGET & SAVINGS", value="budget",
                style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label="RISK & FAILURE", value="risk",
                style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label="TIMELINE",       value="timeline",
                style=tab_style, selected_style=tab_selected_style),
    ], style={"borderBottom": "1px solid #1C2D3D", "marginBottom": "20px"}),

    html.Div(id="tab-content"),
])


# ────────────────────────────────────────────────────────────
# 6.  CALLBACKS
# ────────────────────────────────────────────────────────────

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("bac-filter", "value"),
    Input("sem-filter", "value"),
)
def render_tab(tab, bac_val, sem_val):
    # Apply filters
    fdf = df.copy()
    if bac_val != "ALL":
        fdf = fdf[fdf["bac_short"] == bac_val]
    if sem_val != "ALL":
        fdf = fdf[fdf["semester"] == sem_val]

    fc = fdf[fdf["status"] == "Completed"].copy()
    ff = fdf[fdf["status"] == "Failed"].copy()

    def _chart_layout(title):
        return chart_layout(title)

    # ── OVERVIEW ────────────────────────────────────────────
    if tab == "overview":
        # Status donut
        counts = fdf["status"].value_counts()
        donut = go.Figure(go.Pie(
            labels=counts.index, values=counts.values, hole=0.62,
            marker_colors=[STATUS_COLORS.get(s, "#95A5A6") for s in counts.index],
            textinfo="label+percent",
            textfont=dict(family=MONO, size=11, color=TEXT),
        ))
        donut.update_layout(**_chart_layout("Status Breakdown"))
        donut.update_layout(annotations=[dict(
            text=f"<b>{len(fdf)}</b><br><span style='font-size:9px'>contracts</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=17, color=TEXT, family=MONO),
        )])

        # BAC bar
        pivot = fdf.groupby(["bac_short", "status"]).size().reset_index(name="count")
        bac_bar = px.bar(
            pivot, x="count", y="bac_short", color="status",
            color_discrete_map=STATUS_COLORS,
            orientation="h", barmode="stack",
            labels={"count": "Contracts", "bac_short": "", "status": "Status"},
        )
        bac_bar.update_layout(**_chart_layout("Contracts by BAC Unit"))
        bac_bar.update_traces(marker_line_width=0)

        return html.Div([
            html.Div([
                dcc.Graph(figure=donut, config=graph_config,
                          style={"flex": "1", "minWidth": "300px"}),
                dcc.Graph(figure=bac_bar, config=graph_config,
                          style={"flex": "2", "minWidth": "400px"}),
            ], style={"display": "flex", "gap": "16px"}),
        ])

    # ── BUDGET & SAVINGS ────────────────────────────────────
    elif tab == "budget":
        if fc.empty:
            return html.P("No completed contracts match the current filter.",
                          style={"color": "#8899AA", "padding": "40px"})

        fc2 = fc.copy()
        fc2["project_short"] = fc2["project"].str[:42] + "…"
        fc2 = fc2.sort_values("abc_php", ascending=True)

        overlap_bar = go.Figure()
        overlap_bar.add_trace(go.Bar(
            y=fc2["project_short"], x=fc2["abc_php"],
            name="ABC", orientation="h",
            marker_color="#3498DB33",
            marker_line_color="#3498DB", marker_line_width=1.2,
        ))
        overlap_bar.add_trace(go.Bar(
            y=fc2["project_short"], x=fc2["contract_php"],
            name="Contract Price", orientation="h",
            marker_color="#2ECC71CC",
        ))
        overlap_bar.update_layout(**_chart_layout("ABC vs Contract Price — Completed Contracts"))
        overlap_bar.update_layout(
            barmode="overlay",
            xaxis_tickprefix="₱", xaxis_tickformat=",.0f",
            height=max(350, len(fc2) * 30),
        )

        fc2["project_hover"] = fc2["project"].str[:55]
        scatter = px.scatter(
            fc2, x="abc_php", y="savings_pct",
            size="abc_php", color="bac_short",
            hover_name="project_hover",
            hover_data={"abc_php": ":,.0f", "savings_pct": ":.2f", "bac_short": False},
            labels={"abc_php": "ABC (₱)", "savings_pct": "Savings %", "bac_short": "BAC"},
            size_max=40,
        )
        scatter.update_layout(**_chart_layout("ABC vs Savings % (bubble = contract size)"))
        scatter.update_layout(xaxis_tickprefix="₱", xaxis_tickformat=",.0f")
        scatter.update_traces(marker_line_width=0.5, marker_line_color="#ECEFF4")

        return html.Div([
            dcc.Graph(figure=overlap_bar, config=graph_config),
            html.Div(style={"height": "16px"}),
            dcc.Graph(figure=scatter, config=graph_config),
        ])

    # ── RISK & FAILURE ───────────────────────────────────────
    elif tab == "risk":
        bad = fdf[fdf["status"].isin(["Failed", "Cancelled"])]

        if bad.empty:
            return html.P("No failed or cancelled contracts match the current filter.",
                          style={"color": "#8899AA", "padding": "40px"})

        pivot = bad.groupby(["bac_short", "mode"]).size().unstack(fill_value=0)
        heatmap = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0, "#0F1923"], [0.4, "#E67E2299"], [1, "#E74C3C"]],
            text=pivot.values,
            texttemplate="%{text}",
            textfont=dict(color=TEXT, size=12, family=MONO),
            hovertemplate="BAC: %{y}<br>Mode: %{x}<br>Count: %{z}<extra></extra>",
        ))
        heatmap.update_layout(**_chart_layout("Failure & Cancellation — BAC × Procurement Mode"))
        heatmap.update_layout(xaxis_tickangle=-25)

        reasons_fig = go.Figure()
        if "failure_reason" in ff.columns:
            reasons = ff["failure_reason"].dropna().value_counts()
            if not reasons.empty:
                reasons_fig = px.bar(
                    x=reasons.values, y=reasons.index,
                    orientation="h",
                    labels={"x": "Count", "y": ""},
                    color=reasons.values,
                    color_continuous_scale=["#E67E22", "#E74C3C"],
                )
                reasons_fig.update_layout(**_chart_layout("Failure Reasons (RA 9184 Sections)"))
                reasons_fig.update_coloraxes(showscale=False)
                reasons_fig.update_traces(marker_line_width=0)

        return html.Div([
            html.Div([
                dcc.Graph(figure=heatmap, config=graph_config,
                          style={"flex": "2"}),
                dcc.Graph(figure=reasons_fig, config=graph_config,
                          style={"flex": "1"}),
            ], style={"display": "flex", "gap": "16px"}),
        ])

    # ── TIMELINE ────────────────────────────────────────────
    elif tab == "timeline":
        tc = fc.dropna(subset=["pre_proc_conference", "notice_to_proceed"]).copy()
        if tc.empty:
            return html.P("No completed contracts with full date data match the filter.",
                          style={"color": "#8899AA", "padding": "40px"})

        tc = tc.sort_values("pre_proc_conference")
        tc["project_short"] = tc["project"].str[:38] + "…"

        STAGES = [
            ("pre_proc_conference",    "posting_ib",               "Pre-Proc → IB",     "#3498DB"),
            ("posting_ib",             "pre_bid_conference",        "IB → Pre-Bid",      "#1ABC9C"),
            ("pre_bid_conference",     "submission_opening_bids",   "Pre-Bid → Open",    "#F39C12"),
            ("submission_opening_bids","bac_resolution",            "Open → Resolution", "#9B59B6"),
            ("bac_resolution",         "contract_signing",          "Resolution → Sign", "#E74C3C"),
            ("contract_signing",       "notice_to_proceed",         "Sign → NTP",        "#2ECC71"),
        ]

        gantt = go.Figure()
        for row_i, (_, row) in enumerate(tc.iterrows()):
            for start_col, end_col, stage_name, color in STAGES:
                s = row.get(start_col)
                e = row.get(end_col)
                if pd.notna(s) and pd.notna(e) and e >= s:
                    gantt.add_trace(go.Bar(
                        x=[(e - s).days],
                        base=[s.timestamp() * 1000],
                        y=[row["project_short"]],
                        orientation="h",
                        marker_color=color,
                        name=stage_name,
                        legendgroup=stage_name,
                        showlegend=(row_i == 0),
                        hovertemplate=(
                            f"<b>{row['project_short']}</b><br>"
                            f"Stage: {stage_name}<br>"
                            f"From: {s.date()} → {e.date()}<br>"
                            f"Duration: {(e-s).days} days<extra></extra>"
                        ),
                    ))

        gantt.update_layout(**_chart_layout("Procurement Lifecycle — Completed Contracts"))
        gantt.update_layout(
            barmode="overlay",
            xaxis=dict(type="date", tickformat="%b %Y", tickfont=dict(size=10)),
            height=max(400, len(tc) * 32),
            legend=dict(orientation="h", y=-0.06, font_size=10),
        )

        return html.Div([dcc.Graph(figure=gantt, config=graph_config)])

    return html.Div()


# ────────────────────────────────────────────────────────────
# 7.  RUN
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  DOTr PMR CY 2021 Dashboard")
    print("  Open → http://127.0.0.1:8050")
    print("=" * 55)
    app.run(debug=False)
