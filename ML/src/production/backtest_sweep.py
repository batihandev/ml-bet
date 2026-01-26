import pandas as pd
import numpy as np
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from .backtest_utils import (
    build_df_bt, index_predictions, 
    compute_stake, compute_profit, get_actual_outcome, compute_kelly_fraction
)
from .betting_logic import select_bet, metric_passes_gate
from .predict import predict_ft_1x2
from .schema import CLASS_MAPPING
from .bootstrap import bootstrap_roi
from dataset.cleaner import load_features

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
LATEST_SWEEP_PATH = DATA_DIR / "backtest_sweep_latest.json"

def save_latest_sweep(data: Dict[str, Any]):
    """Save the sweep results to the data folder."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(LATEST_SWEEP_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save sweep results: {e}")

def load_latest_sweep() -> Optional[Dict[str, Any]]:
    """Load the latest sweep results from the data folder."""
    if not LATEST_SWEEP_PATH.exists():
        return None
    try:
        with open(LATEST_SWEEP_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load sweep results: {e}")
        return None

def _apply_blend(metrics: List[Dict[str, Any]], alpha: float) -> List[Dict[str, Any]]:
    if not metrics or alpha >= 0.999:
        return metrics
    blended = []
    for m in metrics:
        if "pimp" not in m or m.get("pimp") is None:
            blended.append(m)
            continue
        prob_model = float(m.get("prob", 0.0))
        prob_implied = float(m.get("pimp", 0.0))
        odds = float(m.get("odds", 0.0))
        prob_blend = (alpha * prob_model) + ((1.0 - alpha) * prob_implied)
        m_new = dict(m)
        m_new["prob_model"] = prob_model
        m_new["prob_implied"] = prob_implied
        m_new["prob"] = prob_blend
        m_new["edge"] = prob_blend - prob_implied
        m_new["ev"] = prob_blend * odds - 1.0
        blended.append(m_new)
    return blended

def _get_group_stats(rows, odds_key="odds", outcome_key="outcome"):
    if not rows:
        return {
            "count": 0, "avg_odds": 0, "med_odds": 0, "p90_odds": 0,
            "mix": {"home": 0, "draw": 0, "away": 0}
        }
    df_g = pd.DataFrame(rows)
    n_g = len(df_g)
    c_g = df_g[outcome_key].value_counts().to_dict()
    return {
        "count": n_g,
        "avg_odds": float(round(df_g[odds_key].mean(), 2)),
        "med_odds": float(round(df_g[odds_key].median(), 2)),
        "p90_odds": float(round(df_g[odds_key].quantile(0.9), 2)),
        "mix": {
            "home": float(round(c_g.get("home", 0) / n_g, 3)),
            "draw": float(round(c_g.get("draw", 0) / n_g, 3)),
            "away": float(round(c_g.get("away", 0) / n_g, 3))
        }
    }

def _run_alpha_sweep_task(args: Tuple[Any, ...]) -> Tuple[str, Dict[str, Any]]:
    (
        alpha,
        base_matches,
        edges,
        evs,
        stake,
        kelly_mult,
        min_bets,
        bootstrap_n,
        max_ci_cells,
        selection_mode,
        alpha_range_used,
        resolved_start,
        resolved_end
    ) = args

    # Build blended match list once per alpha
    processed_matches = []
    n_total_all_valid = 0
    for m in base_matches:
        metrics_used = _apply_blend(m["metrics"], alpha)
        if not metrics_used:
            continue
        top_metric = max(metrics_used, key=lambda x: x["prob"])
        processed_matches.append({
            "id": m["id"],
            "date": m["date"],
            "actual": m["actual"],
            "metrics": metrics_used,
            "top_outcome": top_metric["outcome"],
            "top_odds": top_metric["odds"],
            "top_prob": top_metric["prob"],
            "top_edge": top_metric["edge"],
            "top_ev": top_metric["ev"]
        })
        n_total_all_valid += 1

    cells = []
    for me in edges:
        for mv in evs:
            bet_rows = []
            n_any_passes_gate = 0
            n_top_prob_passes_gate = 0

            # For distributions
            group_top_prob_all_valid = []
            group_top_prob_passes_gate = []

            bankroll = float(stake) if kelly_mult > 0 else None

            for m in processed_matches:
                # ALL_VALID is constant for all cells
                group_top_prob_all_valid.append({
                    "outcome": m["top_outcome"],
                    "odds": m["top_odds"]
                })

                # Check gates
                any_passes = any(metric_passes_gate(met, me, mv, selection_mode) for met in m["metrics"])
                if any_passes:
                    n_any_passes_gate += 1

                top_passes = metric_passes_gate(
                    {
                        "edge": m["top_edge"],
                        "ev": m["top_ev"],
                        "odds": m["top_odds"],
                        "prob": m["top_prob"]
                    },
                    me,
                    mv,
                    selection_mode
                )
                if top_passes:
                    n_top_prob_passes_gate += 1
                    group_top_prob_passes_gate.append({
                        "outcome": m["top_outcome"],
                        "odds": m["top_odds"]
                    })

                # Selection logic
                best = select_bet(m["metrics"], me, mv, selection_mode)
                if not best:
                    continue

                is_win = (m["actual"] == CLASS_MAPPING[best["outcome"]])

                if kelly_mult > 0:
                    if bankroll is None or bankroll <= 0:
                        continue
                    kelly_f = compute_kelly_fraction(best["prob"], best["odds"])
                    s = bankroll * kelly_mult * kelly_f
                    if s <= 0:
                        continue
                    if s > bankroll:
                        s = bankroll
                else:
                    s = compute_stake(stake, kelly_mult, best["prob"], best["odds"])
                    if s <= 0:
                        continue

                p = compute_profit(s, best["odds"], is_win)
                if kelly_mult > 0 and bankroll is not None:
                    bankroll += p
                bet_rows.append({
                    "date": m["date"],
                    "stake": s,
                    "profit": p,
                    "odds": best["odds"],
                    "ev": best["ev"],
                    "edge": best["edge"],
                    "prob": best["prob"],
                    "outcome": best["outcome"]
                })

            stats_all_valid = _get_group_stats(group_top_prob_all_valid)
            stats_top_passes = _get_group_stats(group_top_prob_passes_gate)
            stats_placed = _get_group_stats(bet_rows, odds_key="odds", outcome_key="outcome")

            all_valid_definition = "prediction present + labeled outcome + valid odds"

            if not bet_rows:
                cells.append({
                    "min_edge": float(me), "min_ev": float(mv),
                    "bets": 0, "roi": 0.0, "profit": 0.0, "low_sample": True,
                    "n_all_valid": n_total_all_valid,
                    "n_all_valid_matches": n_total_all_valid,
                    "n_any_passes_gate": n_any_passes_gate,
                    "n_top_prob_passes_gate": n_top_prob_passes_gate,
                    "stats_all_valid": stats_all_valid,
                    "stats_top_prob_all_valid": stats_all_valid,
                    "stats_top_passes_gate": stats_top_passes,
                    "stats_placed_bets": stats_placed,
                    "all_valid_definition": all_valid_definition,
                    "pct_h": stats_placed["mix"]["home"],
                    "pct_d": stats_placed["mix"]["draw"],
                    "pct_a": stats_placed["mix"]["away"],
                })
                continue

            bdf = pd.DataFrame(bet_rows)
            staked = bdf["stake"].sum()
            profit = bdf["profit"].sum()
            bets = len(bdf)

            cell = {
                "min_edge": float(me),
                "min_ev": float(mv),
                "bets": bets,
                "roi": float(round(profit / staked, 4)) if staked > 0 else 0.0,
                "profit": float(round(profit, 2)),
                "avg_odds": stats_placed["avg_odds"],
                "median_odds": stats_placed["med_odds"],
                "p90_odds": stats_placed["p90_odds"],
                "avg_ev": float(round(bdf["ev"].mean(), 4)),
                "avg_edge": float(round(bdf["edge"].mean(), 4)),
                "low_sample": bool(bets < min_bets),

                # Diagnostics
                "n_all_valid": n_total_all_valid,
                "n_all_valid_matches": n_total_all_valid,
                "n_any_passes_gate": n_any_passes_gate,
                "n_top_prob_passes_gate": n_top_prob_passes_gate,

                "stats_all_valid": stats_all_valid,
                "stats_top_prob_all_valid": stats_all_valid,
                "stats_top_passes_gate": stats_top_passes,
                "stats_placed_bets": stats_placed,
                "all_valid_definition": all_valid_definition,

                # Backward compatibility for existing UI fields
                "pct_h": stats_placed["mix"]["home"],
                "pct_d": stats_placed["mix"]["draw"],
                "pct_a": stats_placed["mix"]["away"],
            }

            cell["_raw_bdf"] = bdf
            cells.append(cell)

    cells.sort(key=lambda x: x["roi"], reverse=True)

    ci_computed = 0
    for cell in cells:
        if cell["bets"] >= min_bets and ci_computed < max_ci_cells:
            ci_res = bootstrap_roi(cell["_raw_bdf"], n=bootstrap_n)
            if ci_res["status"] == "success":
                cell["roi_p05"] = float(round(ci_res["roi_p05"], 4))
                cell["roi_p95"] = float(round(ci_res["roi_p95"], 4))
                cell["profit_p05"] = float(round(ci_res["profit_p05"], 2))
                cell["profit_p95"] = float(round(ci_res["profit_p95"], 2))
                ci_computed += 1

        if "_raw_bdf" in cell:
            del cell["_raw_bdf"]

    cells.sort(key=lambda x: (x.get("roi_p05", -999), x["roi"]), reverse=True)

    summary = {
        "total_matches": len(processed_matches),
        "all_valid_definition": "prediction present + labeled outcome + valid odds",
        "edge_range": (float(edges[0]), float(edges[-1]), float(edges[1] - edges[0])) if len(edges) > 1 else (float(edges[0]), float(edges[0]), 0.0),
        "ev_range": (float(evs[0]), float(evs[-1]), float(evs[1] - evs[0])) if len(evs) > 1 else (float(evs[0]), float(evs[0]), 0.0),
        "alpha_range": alpha_range_used,
        "min_bets": min_bets,
        "start_date": str(resolved_start.date()),
        "end_date": str(resolved_end.date()),
        "stake": stake,
        "kelly_mult": kelly_mult,
        "selection_mode": selection_mode,
        "blend_alpha": float(alpha)
    }

    alpha_key = f"{alpha:.3f}".rstrip("0").rstrip(".")
    return alpha_key, {"cells": cells, "summary": summary}

def run_backtest_sweep(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    edge_range: Tuple[float, float, float] = (0.0, 0.10, 0.01),
    ev_range: Tuple[float, float, float] = (0.0, 0.10, 0.01),
    alpha_range: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    stake: float = 1.0,
    kelly_mult: float = 0.0,
    min_bets: int = 300,
    bootstrap_n: int = 500,
    max_ci_cells: int = 50,
    selection_mode: str = "best_ev",
    debug: int = 0,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_every: int = 50,
    n_jobs: int = 0
) -> Dict[str, Any]:
    """
    Perform an optimized grid search over (min_edge, min_ev).
    Predicts once and reuses metrics.
    """
    df_all = load_features()
    
    # Resolve dates
    data_min = pd.to_datetime(df_all["match_date"].min())
    data_max = pd.to_datetime(df_all["match_date"].max())
    resolved_start = pd.to_datetime(start_date) if start_date else data_min
    resolved_end = pd.to_datetime(end_date) if end_date else data_max
    
    df_bt = build_df_bt(df_all, resolved_start, resolved_end)
    if df_bt.empty:
        return {"cells": [], "status": "No matches in window"}

    # Predict once
    predictions = predict_ft_1x2(df_bt, debug=0) # sweep usually doesn't need debug rows per cell
    pred_by_id = index_predictions(predictions)

    # Store base match info (unblended)
    base_matches = []
    for _, match_row in df_bt.iterrows():
        mid = int(match_row["match_id"])
        if mid not in pred_by_id:
            continue
        actual = get_actual_outcome(match_row)
        if actual is None:
            continue
        metrics = pred_by_id[mid]["metrics"]
        if not metrics:
            continue  # Invalid odds
        base_matches.append({
            "id": mid,
            "date": match_row["match_date"],
            "actual": actual,
            "metrics": metrics
        })
    
    # Generate grid
    edges = np.round(np.arange(edge_range[0], edge_range[1] + 0.0001, edge_range[2]), 3)
    evs = np.round(np.arange(ev_range[0], ev_range[1] + 0.0001, ev_range[2]), 3)
    # Generate alpha grid
    try:
        a_start = float(alpha_range[0])
        a_end = float(alpha_range[1])
        a_step = float(alpha_range[2])
    except Exception:
        a_start, a_end, a_step = 1.0, 1.0, 1.0
    if a_step <= 0:
        a_step = 0.1
    a_start = max(0.0, min(1.0, a_start))
    a_end = max(0.0, min(1.0, a_end))
    if a_end < a_start:
        a_start, a_end = a_end, a_start
    alphas = np.round(np.arange(a_start, a_end + 0.0001, a_step), 3)
    if len(alphas) == 0:
        alphas = np.array([1.0])
    alpha_range_used = (float(a_start), float(a_end), float(a_step))
    total_cells = int(len(alphas) * len(edges) * len(evs))
    cell_idx = 0

    alpha_results: Dict[str, Any] = {}
    # Decide parallelism
    if n_jobs is None or n_jobs <= 0:
        cpu = os.cpu_count() or 2
        n_jobs = max(1, min(cpu - 1, len(alphas)))
    use_parallel = n_jobs > 1 and len(alphas) > 1

    if use_parallel:
        # Process per-alpha in parallel (safe with spawn even when called from threads)
        ctx = get_context("spawn")
        tasks = [
            (
                float(alpha),
                base_matches,
                edges,
                evs,
                stake,
                kelly_mult,
                min_bets,
                bootstrap_n,
                max_ci_cells,
                selection_mode,
                alpha_range_used,
                resolved_start,
                resolved_end
            )
            for alpha in alphas
        ]
        done_alphas = 0
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
            futures = [ex.submit(_run_alpha_sweep_task, t) for t in tasks]
            for fut in as_completed(futures):
                alpha_key, payload = fut.result()
                alpha_results[alpha_key] = payload
                done_alphas += 1
                if progress_callback:
                    progress_callback({
                        "done": done_alphas,
                        "total": len(alphas),
                        "pct": float(round(done_alphas / len(alphas), 4)) if len(alphas) > 0 else 1.0,
                        "alpha": alpha_key
                    })
    else:
        done_alphas = 0
        for alpha in alphas:
            alpha_key, payload = _run_alpha_sweep_task((
                float(alpha),
                base_matches,
                edges,
                evs,
                stake,
                kelly_mult,
                min_bets,
                bootstrap_n,
                max_ci_cells,
                selection_mode,
                alpha_range_used,
                resolved_start,
                resolved_end
            ))
            alpha_results[alpha_key] = payload
            done_alphas += 1
            if progress_callback:
                progress_callback({
                    "done": done_alphas,
                    "total": len(alphas),
                    "pct": float(round(done_alphas / len(alphas), 4)) if len(alphas) > 0 else 1.0,
                    "alpha": alpha_key
                })

    default_alpha = f"{alphas[0]:.3f}".rstrip("0").rstrip(".") if len(alphas) else "1"

    result = {
        "alpha_results": alpha_results,
        "default_alpha": default_alpha,
        "summary": {
            "total_matches": len(base_matches),
            "all_valid_definition": "prediction present + labeled outcome + valid odds",
            "edge_range": edge_range,
            "ev_range": ev_range,
            "alpha_range": alpha_range_used,
            "min_bets": min_bets,
            "start_date": str(resolved_start.date()),
            "end_date": str(resolved_end.date()),
            "stake": stake,
            "kelly_mult": kelly_mult,
            "selection_mode": selection_mode
        }
    }

    # Backward-compatible top-level keys for single-alpha sweeps
    if len(alphas) == 1 and default_alpha in alpha_results:
        result["cells"] = alpha_results[default_alpha]["cells"]
        result["summary"] = alpha_results[default_alpha]["summary"]

    save_latest_sweep(result)
    return result
