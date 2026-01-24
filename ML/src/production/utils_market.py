import numpy as np
from typing import Optional, Tuple, Any

def is_valid_odds(x: Any) -> bool:
    """
    Check if an odd value is valid:
    - Finite float
    - Greater than 1.0
    """
    try:
        val = float(x)
        return np.isfinite(val) and val > 1.0
    except (ValueError, TypeError):
        return False

def calculate_implied_probs(odd_home: Any, odd_draw: Any, odd_away: Any) -> Optional[Tuple[float, float, float]]:
    """
    Calculate normalized implied probabilities from bookmaker odds.
    Returns None if any odd is invalid or if margin is non-positive/infinite.
    """
    if not (is_valid_odds(odd_home) and is_valid_odds(odd_draw) and is_valid_odds(odd_away)):
        return None
        
    try:
        oh = float(odd_home)
        od = float(odd_draw)
        oa = float(odd_away)
        
        # Avoid division by zero naturally handled by is_valid_odds > 1.0 check, 
        # but safe to be explicit if needed. Since >1.0, 1/x is safe.
        
        raw_probs = np.array([1.0/oh, 1.0/od, 1.0/oa])
        margin = np.sum(raw_probs)
        
        if not np.isfinite(margin) or margin <= 0:
            return None
            
        probs = raw_probs / margin
        return (float(probs[0]), float(probs[1]), float(probs[2]))
    except Exception:
        return None
