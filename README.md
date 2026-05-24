# DOTr Procurement Risk Optimization & Dashboard (CY 2021)

This project analyzes the Department of Transportation (DOTr) Procurement Monitoring Report (PMR) for CY 2021. It features an interactive visualization dashboard and a prescriptive analytics model that optimizes procurement portfolios to minimize failure risks.

## Project Overview

* **Data Engineering (`dotr_pmr_2021_data.py`):** Cleans and standardizes raw procurement data, extracting key variables such as the Approved Budget for the Contract (ABC), contract prices, and procurement modes.
* **Interactive Dashboard (`dotr_pmr_2021_dashboard.py`):** A Plotly Dash application providing a high-level overview of total contracts, budget utilization, savings, and historical failure rates across different Bids and Awards Committees (BAC).
* **Portfolio Optimization (`dotr_risk_minimization.py`):** A linear programming model built with PuLP. It prescribes an optimal combination of contracts to fund under specific budget constraints while minimizing the aggregate risk of failure or cancellation.

## The Optimization Model (Prescriptive Analytics)

The integer linear programming (ILP) model selects a subset of contracts to minimize overall procurement risk.

**Decision Variable:**
Let $x_i \in \{0, 1\}$ represent whether contract $i$ is included in the approved portfolio ($1$) or deferred ($0$).

**Objective Function:**
Minimize the total weighted risk score of the selected portfolio. The risk score accounts for procurement mode competitiveness, historical BAC failure rates, contract size, and projected savings quality.

**Constraints:**
1. **Global Budget Cap:** The sum of the ABC for selected contracts must not exceed 80% of the total universe budget.
2. **Per-BAC Limit:** The allocated budget for any single BAC unit cannot exceed 85% of its total historical ABC, forcing the exclusion of high-risk outlier contracts.
3. **Savings Threshold:** Contracts must meet a minimum effective savings threshold to be eligible for selection.

## Tech Stack

* **Language:** Python 3.10+
* **Data Processing:** pandas
* **Visualization:** Plotly, Dash
* **Optimization/Linear Programming:** PuLP (CBC Solver)

## Installation & Usage

1. Clone this repository:
   git clone https://github.com/your-username/dotr-pmr-analytics.git
   cd dotr-pmr-analytics

2. Install dependencies:
   pip install -r requirements.txt

3. Run the interactive dashboard:
   python src/dotr_pmr_2021_dashboard.py
   Open http://127.0.0.1:8050 in your web browser.

4. Run the risk minimization model:
   python src/dotr_risk_minimization.py
   This will output the optimization results to the console and generate the `dotr_pmr_2021_optimized_portfolio.csv` export.
