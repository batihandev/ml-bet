import unittest
from production.betting_logic import select_bet

class TestBettingLogic(unittest.TestCase):
    def test_best_ev_mode(self):
        # Case: multiple outcomes pass, choose max EV
        metrics = [
            {"outcome": "home", "prob": 0.5, "odds": 2.2, "edge": 0.05, "ev": 0.10},  # Valid
            {"outcome": "draw", "prob": 0.2, "odds": 6.0, "edge": 0.10, "ev": 0.20},  # Valid, Max EV
            {"outcome": "away", "prob": 0.3, "odds": 2.0, "edge": -0.05, "ev": -0.4} # Invalid
        ]
        
        # Should pick draw (max EV)
        bet = select_bet(metrics, min_edge=0.01, min_ev=0.01, selection_mode="best_ev")
        self.assertIsNotNone(bet)
        self.assertEqual(bet["outcome"], "draw")

    def test_top_prob_only_success(self):
        # Case: top probability passes thresholds
        metrics = [
            {"outcome": "home", "prob": 0.6, "odds": 2.0, "edge": 0.10, "ev": 0.20},  # Top Prob, Valid
            {"outcome": "draw", "prob": 0.2, "odds": 6.0, "edge": 0.10, "ev": 0.20},  # Lower Prob
            {"outcome": "away", "prob": 0.2, "odds": 2.0, "edge": -0.10, "ev": -0.10}
        ]
        
        bet = select_bet(metrics, min_edge=0.05, min_ev=0.05, selection_mode="top_prob_only")
        self.assertIsNotNone(bet)
        self.assertEqual(bet["outcome"], "home")

    def test_top_prob_only_failure(self):
        # Case: top probability fails thresholds, even if others pass
        metrics = [
            {"outcome": "home", "prob": 0.6, "odds": 1.5, "edge": -0.05, "ev": -0.10}, # Top Prob, Invalid
            {"outcome": "draw", "prob": 0.3, "odds": 4.0, "edge": 0.10, "ev": 0.20},   # Lower Prob, Valid
            {"outcome": "away", "prob": 0.1, "odds": 1.1, "edge": -0.20, "ev": -0.5}
        ]
        
        # top_prob_only should return None because "home" fails
        bet = select_bet(metrics, min_edge=0.01, min_ev=0.01, selection_mode="top_prob_only")
        self.assertIsNone(bet)
        
        # best_ev should return "draw"
        bet_ev = select_bet(metrics, min_edge=0.01, min_ev=0.01, selection_mode="best_ev")
        self.assertEqual(bet_ev["outcome"], "draw")

    def test_empty_metrics(self):
        self.assertIsNone(select_bet([], 0, 0))

    def test_invalid_odds(self):
        metrics = [{"outcome": "home", "prob": 0.9, "odds": 1.0, "edge": 0.8, "ev": -0.1}]
        self.assertIsNone(select_bet(metrics, 0, 0))

if __name__ == "__main__":
    unittest.main()
