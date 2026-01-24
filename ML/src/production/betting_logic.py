from typing import Optional, Dict, Any, List

def select_bet(
    metrics: List[Dict[str, Any]], 
    min_edge: float, 
    min_ev: float, 
    selection_mode: str = "best_ev"
) -> Optional[Dict[str, Any]]:
    """
    Apply gating and selection logic based on the chosen mode.
    
    Supported selection_modes:
    - "best_ev": among outcomes that pass min_edge and min_ev, select the one with max ev.
    - "top_prob_only": identify the outcome with the highest probability. 
      Only bet it if it passes min_edge and min_ev; otherwise return None.
    """
    if not metrics:
        return None

    if selection_mode == "top_prob_only":
        # 1. Pick the outcome with max probability first
        top_metric = max(metrics, key=lambda x: x["prob"])
        
        # 2. Gate it
        if (top_metric["edge"] >= min_edge and 
            top_metric["ev"] >= min_ev and 
            top_metric["odds"] > 1.0):
            return top_metric
        return None

    else:  # default to "best_ev"
        # 1. Gate everything first
        valid_bets = [
            m for m in metrics 
            if m["edge"] >= min_edge and m["ev"] >= min_ev and m["odds"] > 1.0
        ]
        
        if not valid_bets:
            return None
            
        # 2. Pick max ev among valid
        return max(valid_bets, key=lambda x: x["ev"])
