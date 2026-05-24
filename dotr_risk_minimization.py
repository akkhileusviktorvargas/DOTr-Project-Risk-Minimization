"""
DOTr PMR CY 2021 — Procurement Portfolio Optimization Model
============================================================
Objective  : Minimize total weighted risk score of the selected procurement
             portfolio (i.e. pick the combination of contracts that carries
             the least risk of failure/cancellation).

Decision   : For each contract i, binary variable x[i] ∈ {0, 1}
             x[i] = 1  →  include contract i in the approved portfolio
             x[i] = 0  →  exclude (defer / descope)

Constraints:
  C1. Fixed total budget cap    — sum of selected ABC ≤ TOTAL_BUDGET_CAP
  C2. Per-BAC unit budget limit — sum of selected ABC within each BAC ≤ BAC_LIMIT[bac]
  C3. Savings threshold         — only contracts whose historical savings %
                                  (or expected savings proxy) ≥ MIN_SAVINGS_PCT
                                  are eligible for selection

Risk score per contract is composed of:
  • Base risk by procurement mode        (Public Bidding is riskier than DC/Negotiated)
  • Failure-rate penalty per BAC unit    (derived from historical data)
  • ABC size penalty                     (large contracts are inherently riskier)
  • Savings quality bonus                (high savings % lowers risk)

Solver: PuLP (CBC) — pure Python, no commercial license needed.
        Install: pip install pulp pandas

Author  : Generated from DOTr PMR CY 2021 data
Date    : 2025
"""

import sys
import pandas as pd
import pulp

# ---------------------------------------------------------------------------
# IMPORT DATA  (works both when run standalone or alongside the data module)
# ---------------------------------------------------------------------------
from dotr_pmr_2021_data import records, build_dataframe
# ---------------------------------------------------------------------------
# 1.  BUILD & CLEAN DATAFRAME
# ---------------------------------------------------------------------------

df = build_dataframe(records)

# Keep only contracts with a known ABC (can't budget without it)
df = df[df["abc_php"].notna()].copy()
df = df.reset_index(drop=True)

print(f"Total contracts with known ABC : {len(df)}")
print(f"Status breakdown:\n{df['status'].value_counts().to_string()}\n")

# ---------------------------------------------------------------------------
# 2.  OPTIMIZATION PARAMETERS  (tweak here to explore scenarios)
# ---------------------------------------------------------------------------

# C1 — Global budget cap: 80 % of total ABC across all records
TOTAL_BUDGET_CAP: float = df["abc_php"].sum() * 0.80

# C2 — Per-BAC cap: 85 % of that BAC's total ABC
#       (forces each unit to shed its riskiest tail)
BAC_BUDGET_RATIO: float = 0.85

# C3 — Minimum savings threshold (% of ABC)
#       Contracts with savings_pct < this value are ineligible
#       For contracts without a final contract price (On-Going / Failed /
#       Cancelled), we use a mode-based proxy (see below).
MIN_SAVINGS_PCT: float = 0.0          # ≥ 0 % savings required (adjust upward to tighten)

# Risk weight components (all between 0-1, tunable)
W_MODE    = 0.35   # weight for procurement-mode risk
W_BAC     = 0.30   # weight for BAC historical failure rate
W_SIZE    = 0.20   # weight for contract size (normalised)
W_SAVINGS = 0.15   # weight for savings quality (inverse)

# ---------------------------------------------------------------------------
# 3.  FEATURE ENGINEERING — RISK COMPONENTS
# ---------------------------------------------------------------------------

# 3a. Procurement-mode risk (based on RA 9184 competitiveness spectrum)
MODE_RISK = {
    "Public Bidding"                        : 0.70,
    "International Competitive Bidding"     : 0.80,
    "Limited Competitive Bidding"           : 0.65,
    "Negotiated - Media Services"           : 0.20,
    "Negotiated - HTC"                      : 0.25,
    "Negotiated Procurement"                : 0.30,
    "Direct Contracting"                    : 0.10,
}
DEFAULT_MODE_RISK = 0.55

df["mode_risk"] = df["mode"].map(MODE_RISK).fillna(DEFAULT_MODE_RISK)

# 3b. BAC historical failure rate  (failed + cancelled) / total per BAC
bac_stats = (
    df.groupby("bac")["status"]
    .apply(lambda s: (s.isin(["Failed", "Cancelled"])).sum() / len(s))
    .rename("bac_failure_rate")
)
df = df.join(bac_stats, on="bac")

# 3c. Normalised contract size (0 = smallest, 1 = largest)
abc_min, abc_max = df["abc_php"].min(), df["abc_php"].max()
df["size_risk"] = (df["abc_php"] - abc_min) / (abc_max - abc_min + 1e-9)

# 3d. Savings quality
#     For completed contracts: use actual savings_pct (clipped 0-100)
#     For non-completed: assign mode-based proxy savings expectation
MODE_PROXY_SAVINGS = {
    "Public Bidding"                    : 10.0,
    "International Competitive Bidding" : 8.0,
    "Limited Competitive Bidding"       : 9.0,
    "Negotiated - Media Services"       : 2.0,
    "Negotiated - HTC"                  : 5.0,
    "Negotiated Procurement"            : 3.0,
    "Direct Contracting"                : 0.0,
}
DEFAULT_PROXY = 5.0

def effective_savings_pct(row: pd.Series) -> float:
    if pd.notna(row["savings_pct"]) and row["savings_pct"] >= 0:
        return float(row["savings_pct"])
    return MODE_PROXY_SAVINGS.get(row["mode"], DEFAULT_PROXY)

df["eff_savings_pct"] = df.apply(effective_savings_pct, axis=1)

# Savings risk: high savings → low risk  (invert and normalise 0-1)
max_sav = df["eff_savings_pct"].max() or 1.0
df["savings_risk"] = 1.0 - (df["eff_savings_pct"] / max_sav).clip(0, 1)

# 3e. Composite risk score  (weighted sum, range 0-1)
df["risk_score"] = (
    W_MODE    * df["mode_risk"]       +
    W_BAC     * df["bac_failure_rate"] +
    W_SIZE    * df["size_risk"]        +
    W_SAVINGS * df["savings_risk"]
).round(4)

print("Risk score descriptive stats:")
print(df["risk_score"].describe().round(4).to_string())
print()

# ---------------------------------------------------------------------------
# 4.  ELIGIBILITY FILTER  (Constraint C3)
# ---------------------------------------------------------------------------

df["eligible"] = df["eff_savings_pct"] >= MIN_SAVINGS_PCT

n_eligible   = df["eligible"].sum()
n_ineligible = (~df["eligible"]).sum()
print(f"Eligible contracts   : {n_eligible}")
print(f"Ineligible (savings) : {n_ineligible}\n")

# ---------------------------------------------------------------------------
# 5.  BUILD THE ILP MODEL
# ---------------------------------------------------------------------------

model = pulp.LpProblem("DOTr_Procurement_Risk_Minimization", pulp.LpMinimize)

# Decision variables — one per contract row
x = {
    i: pulp.LpVariable(f"x_{i}", cat="Binary")
    for i in df.index
}

# Force ineligible contracts to 0
for i in df[~df["eligible"]].index:
    model += (x[i] == 0, f"ineligible_{i}")

# ---------- OBJECTIVE ----------
model += (
    pulp.lpSum(df.loc[i, "risk_score"] * x[i] for i in df.index),
    "minimize_total_risk"
)

# ---------- C1: Global budget cap ----------
model += (
    pulp.lpSum(df.loc[i, "abc_php"] * x[i] for i in df.index) <= TOTAL_BUDGET_CAP,
    "global_budget_cap"
)

# ---------- C2: Per-BAC budget limits ----------
for bac_name, bac_df in df.groupby("bac"):
    bac_total   = bac_df["abc_php"].sum()
    bac_cap     = bac_total * BAC_BUDGET_RATIO
    bac_indices = bac_df.index.tolist()
    model += (
        pulp.lpSum(df.loc[i, "abc_php"] * x[i] for i in bac_indices) <= bac_cap,
        f"bac_cap__{bac_name.replace(' ', '_').replace('-', '')}",
    )

# ---------- Minimum selection: at least 1 contract per BAC (optional realism) ----------
for bac_name, bac_df in df.groupby("bac"):
    eligible_in_bac = bac_df[bac_df["eligible"]].index.tolist()
    if eligible_in_bac:
        model += (
            pulp.lpSum(x[i] for i in eligible_in_bac) >= 1,
            f"min_one__{bac_name.replace(' ', '_').replace('-', '')}",
        )

# ---------------------------------------------------------------------------
# 6.  SOLVE
# ---------------------------------------------------------------------------

print("=" * 65)
print("Solving ILP model …")
solver = pulp.PULP_CBC_CMD(msg=False)
status = model.solve(solver)
print(f"Solver status : {pulp.LpStatus[model.status]}")
print("=" * 65)

# ---------------------------------------------------------------------------
# 7.  RESULTS
# ---------------------------------------------------------------------------

df["selected"] = df.index.map(lambda i: int(round(pulp.value(x[i]) or 0)))

selected = df[df["selected"] == 1].copy()
excluded = df[df["selected"] == 0].copy()

total_risk_selected  = selected["risk_score"].sum()
total_risk_all       = df["risk_score"].sum()
total_abc_selected   = selected["abc_php"].sum()
total_abc_all        = df["abc_php"].sum()
avg_risk_selected    = selected["risk_score"].mean() if len(selected) else 0
avg_risk_all         = df["risk_score"].mean()

print(f"\n{'='*65}")
print("OPTIMIZATION RESULTS")
print(f"{'='*65}")
print(f"Contracts in universe         : {len(df)}")
print(f"Contracts SELECTED            : {len(selected)}")
print(f"Contracts EXCLUDED            : {len(excluded)}")
print()
print(f"Total ABC — full universe     : PHP {total_abc_all:>20,.2f}")
print(f"Total ABC — selected          : PHP {total_abc_selected:>20,.2f}")
print(f"Budget cap (C1)               : PHP {TOTAL_BUDGET_CAP:>20,.2f}")
print(f"Budget utilisation            : {total_abc_selected/TOTAL_BUDGET_CAP*100:.1f}%")
print()
print(f"Avg risk score — universe     : {avg_risk_all:.4f}")
print(f"Avg risk score — selected     : {avg_risk_selected:.4f}")
print(f"Risk reduction achieved       : {(1 - avg_risk_selected/avg_risk_all)*100:.1f}%")
print(f"Sum risk — selected portfolio : {total_risk_selected:.4f}")
print(f"{'='*65}\n")

# ---------------------------------------------------------------------------
# 8.  PER-BAC SUMMARY
# ---------------------------------------------------------------------------

print("PER-BAC PORTFOLIO BREAKDOWN")
print("-" * 65)
bac_summary = (
    selected.groupby("bac")
    .agg(
        contracts=("code", "count"),
        total_abc=("abc_php", "sum"),
        avg_risk=("risk_score", "mean"),
        completed=("status", lambda s: (s == "Completed").sum()),
    )
    .sort_values("avg_risk")
)
print(bac_summary.to_string())
print()

# ---------------------------------------------------------------------------
# 9.  TOP-10 HIGHEST-RISK CONTRACTS EXCLUDED
# ---------------------------------------------------------------------------

print("TOP-10 HIGHEST-RISK CONTRACTS EXCLUDED FROM PORTFOLIO")
print("-" * 65)
top_excluded = (
    excluded[excluded["eligible"]]
    .nlargest(10, "risk_score")[
        ["bac", "code", "project", "abc_php", "risk_score", "status", "mode"]
    ]
)
pd.set_option("display.max_colwidth", 45)
pd.set_option("display.width", 200)
print(top_excluded.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 10.  FULL SELECTED PORTFOLIO (sorted by risk ascending)
# ---------------------------------------------------------------------------

print("FULL SELECTED PORTFOLIO  (sorted by risk score ↑)")
print("-" * 65)
portfolio_view = selected[
    ["bac", "code", "project", "abc_php", "eff_savings_pct",
     "risk_score", "status", "mode"]
].sort_values("risk_score")

print(portfolio_view.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 11.  EXPORT TO CSV
# ---------------------------------------------------------------------------

out_path = "dotr_pmr_2021_optimized_portfolio.csv"
portfolio_view.to_csv(out_path, index=False)
print(f"[SAVED] Optimized portfolio → {out_path}")

# ---------------------------------------------------------------------------
# 12.  SENSITIVITY: what happens if we tighten the savings threshold?
# ---------------------------------------------------------------------------

print("\nSENSITIVITY ANALYSIS — varying MIN_SAVINGS_PCT")
print("-" * 65)
print(f"{'Min Savings %':>14} | {'Selected':>8} | {'Avg Risk':>9} | {'Total ABC (B)':>13}")
print("-" * 65)

for threshold in [0, 2, 5, 8, 10, 15]:
    elig_mask = df["eff_savings_pct"] >= threshold
    n_elig    = elig_mask.sum()
    if n_elig == 0:
        print(f"{threshold:>14}% | {'N/A':>8} | {'N/A':>9} | {'N/A':>13}")
        continue

    m2 = pulp.LpProblem(f"Risk_Min_sav{threshold}", pulp.LpMinimize)
    y  = {i: pulp.LpVariable(f"y_{i}", cat="Binary") for i in df.index}

    for i in df[~elig_mask].index:
        m2 += (y[i] == 0)

    m2 += pulp.lpSum(df.loc[i, "risk_score"] * y[i] for i in df.index)
    m2 += pulp.lpSum(df.loc[i, "abc_php"]    * y[i] for i in df.index) <= TOTAL_BUDGET_CAP

    for bac_name, bac_df in df.groupby("bac"):
        bac_cap = bac_df["abc_php"].sum() * BAC_BUDGET_RATIO
        m2 += pulp.lpSum(df.loc[i, "abc_php"] * y[i] for i in bac_df.index) <= bac_cap

    m2.solve(pulp.PULP_CBC_CMD(msg=False))

    sel_idx  = [i for i in df.index if round(pulp.value(y[i]) or 0) == 1]
    sel_df   = df.loc[sel_idx]
    avg_r    = sel_df["risk_score"].mean() if len(sel_df) else 0
    total_a  = sel_df["abc_php"].sum() / 1e9
    print(f"{threshold:>14}% | {len(sel_df):>8} | {avg_r:>9.4f} | {total_a:>13.3f}B")

print("-" * 65)
print("\n[DONE] Optimization model complete.")