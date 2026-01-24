from typing import Optional, Dict, Any, List

def metric_passes_gate(
    metric: Dict[str, Any],
    min_edge: float,
    min_ev: float,
    selection_mode: str = "best_ev"
) -> bool:
    """
    Gate a single metric based on thresholds and selection mode.

    For top_prob_only:
    - Always require edge >= min_edge.
    - Only enforce EV if min_ev > 0. This reduces bias toward long odds when min_ev == 0.

    For top_prob_always:
    - Ignore edge/EV thresholds and accept any valid odds.
    """
    if not metric:
        return False

    odds = float(metric.get("odds", 0.0))
    if odds <= 1.0:
        return False

    edge = float(metric.get("edge", -999.0))
    ev = float(metric.get("ev", -999.0))

    if selection_mode == "top_prob_always":
        return True

    if selection_mode == "top_prob_only":
        if edge < min_edge:
            return False
        if min_ev > 0 and ev < min_ev:
            return False
        return True

    return edge >= min_edge and ev >= min_ev

def select_bet(
    metrics: List[Dict[str, Any]], 
    min_edge: float, 
    min_ev: float, 
    selection_mode: str = "best_ev"
) -> Optional[Dict[str, Any]]:
    """
    Apply gating and selection logic based on the chosen mode.
    
    Supported selection_modes:
    - "best_ev": Among outcomes that pass min_edge and min_ev, select the one with max EV.
    - "top_prob": Among outcomes that pass min_edge and min_ev, select the one with max probability.
    - "top_prob_only": Consider ONLY the outcome with the highest probability.
      Uses edge gate always; EV gate is applied only if min_ev > 0.
    - "top_prob_always": Always take the highest probability outcome (ignores min_edge/min_ev).
    """
    if not metrics:
        return None

    if selection_mode == "top_prob":
        # Filter first, then pick most likely
        valid_bets = [
            m for m in metrics 
            if metric_passes_gate(m, min_edge, min_ev, selection_mode)
        ]
        if not valid_bets:
            return None
        return max(valid_bets, key=lambda x: x["prob"])

    elif selection_mode == "top_prob_only":
        # Absolute most likely outcome only
        top_metric = max(metrics, key=lambda x: x["prob"])
        if metric_passes_gate(top_metric, min_edge, min_ev, selection_mode):
            return top_metric
        return None

    elif selection_mode == "top_prob_always":
        # Always take most likely outcome (odds validity checked in gate helper)
        top_metric = max(metrics, key=lambda x: x["prob"])
        if metric_passes_gate(top_metric, min_edge, min_ev, selection_mode):
            return top_metric
        return None

    else:  # default to "best_ev"
        valid_bets = [
            m for m in metrics 
            if metric_passes_gate(m, min_edge, min_ev, selection_mode)
        ]
        if not valid_bets:
            return None
        return max(valid_bets, key=lambda x: x["ev"])
