"""
Cost of the pass / review / fail decision, shared by train.ipynb and app.ipynb.

Each tile gets a score: the highest confidence of any defect the model finds
in it (0 if it finds none). Two thresholds split the scores three ways:

    score <  t_low            -> pass   (fabric ships)
    t_low <= score < t_high   -> review (a person checks the tile)
    score >= t_high           -> fail   (fabric is pulled)

With t_low == t_high there is no review band and the model decides alone.

The test set has far more defective tiles than a production line would, so
costs are not averaged over test tiles directly. Instead the rates measured on
test (how often a defective tile passes, how often a clean one fails, ...) are
reweighted by the base rate the student assumes for production.

A person who reviews a tile is assumed to get it right: a defective tile they
see is caught (the true-positive cost applies) and a clean one is released
(the true-negative cost applies), on top of the cost of their time.
"""

import numpy as np

OUTCOMES = ["pass", "review", "fail"]


def decide(scores: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    """Score per tile -> 'pass' / 'review' / 'fail'."""
    t_low = min(t_low, t_high)
    return np.where(scores >= t_high, "fail", np.where(scores >= t_low, "review", "pass"))


def outcome_rates(scores: np.ndarray, defect: np.ndarray, t_low: float, t_high: float) -> dict:
    """Share of defective and of clean tiles that end in each outcome."""
    decision = decide(scores, t_low, t_high)
    rates = {}
    for truth, mask in [("defect", defect), ("clean", ~defect)]:
        for outcome in OUTCOMES:
            rates[f"{outcome}|{truth}"] = float((decision[mask] == outcome).mean()) if mask.any() else 0.0
    return rates


def expected_cost(rates: dict, assumptions: dict) -> float:
    """Expected euros per tile in production, at the assumed base rate."""
    cost, base_rate, review = assumptions["cost"], assumptions["base_rate"], assumptions["review_cost"]
    defect = (
        rates["fail|defect"] * cost["tp"]
        + rates["pass|defect"] * cost["fn"]
        + rates["review|defect"] * (review + cost["tp"])
    )
    clean = (
        rates["fail|clean"] * cost["fp"]
        + rates["pass|clean"] * cost["tn"]
        + rates["review|clean"] * (review + cost["tn"])
    )
    return base_rate * defect + (1 - base_rate) * clean


def review_share(rates: dict, base_rate: float) -> float:
    """Share of production tiles sent to a person."""
    return base_rate * rates["review|defect"] + (1 - base_rate) * rates["review|clean"]


def per_thousand(rates: dict, base_rate: float) -> dict:
    """Expected tiles per 1,000 in production for each (truth, outcome) cell."""
    return {
        key: 1000 * rate * (base_rate if key.endswith("|defect") else 1 - base_rate)
        for key, rate in rates.items()
    }


def reviewers_needed(share: float, assumptions: dict) -> float:
    """People needed to clear one day's review queue."""
    hours = share * assumptions["tiles_per_day"] * assumptions["review_seconds"] / 3600
    return hours / assumptions["shift_hours"]


def alternatives(assumptions: dict) -> dict:
    """Expected euros per tile for the designs the model is priced against."""
    cost, base_rate, review = assumptions["cost"], assumptions["base_rate"], assumptions["review_cost"]
    return {
        "do nothing (ship everything)": base_rate * cost["fn"] + (1 - base_rate) * cost["tn"],
        "person checks every tile": review + base_rate * cost["tp"] + (1 - base_rate) * cost["tn"],
    }


def threshold_sweep(scores: np.ndarray, defect: np.ndarray, assumptions: dict, grid: np.ndarray) -> list[dict]:
    """Single threshold (no review band): expected cost at every point on the grid."""
    rows = []
    for t in grid:
        rates = outcome_rates(scores, defect, t, t)
        rows.append({"threshold": float(t), "cost_per_tile": expected_cost(rates, assumptions),
                     "recall": rates["fail|defect"], "false_alarm_rate": rates["fail|clean"]})
    return rows


def best_band(scores: np.ndarray, defect: np.ndarray, assumptions: dict, grid: np.ndarray) -> dict:
    """Cheapest (t_low, t_high) pair on the grid, review band included."""
    best = None
    for i, t_low in enumerate(grid):
        for t_high in grid[i:]:
            rates = outcome_rates(scores, defect, t_low, t_high)
            c = expected_cost(rates, assumptions)
            if best is None or c < best["cost_per_tile"]:
                best = {"t_low": float(t_low), "t_high": float(t_high), "cost_per_tile": c}
    return best
