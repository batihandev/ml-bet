import unittest
import numpy as np
from production.utils_market import is_valid_odds, calculate_implied_probs

class TestMarketUtils(unittest.TestCase):
    def test_is_valid_odds(self):
        # Valid cases
        self.assertTrue(is_valid_odds(1.01))
        self.assertTrue(is_valid_odds(2.5))
        self.assertTrue(is_valid_odds(1000.0))
        self.assertTrue(is_valid_odds("2.5")) # string conversion check
        
        # Invalid cases
        self.assertFalse(is_valid_odds(None))
        self.assertFalse(is_valid_odds(np.nan))
        self.assertFalse(is_valid_odds(0))
        self.assertFalse(is_valid_odds(1.0))
        self.assertFalse(is_valid_odds(0.99))
        self.assertFalse(is_valid_odds(-2.0))
        self.assertFalse(is_valid_odds(np.inf))
        self.assertFalse(is_valid_odds("abc"))

    def test_calculate_implied_probs_valid(self):
        # Balanced book (margin 0, impossible in practice but math works)
        # 2.0, 4.0, 4.0 -> p=0.5, 0.25, 0.25 -> sum=1.0. Normalized is same.
        res = calculate_implied_probs(2.0, 4.0, 4.0)
        self.assertIsNotNone(res)
        self.assertTrue(np.isclose(sum(res), 1.0))
        self.assertTrue(np.allclose(res, [0.5, 0.25, 0.25]))

        # Standard book with margin
        # Odds: 2.0, 3.0, 3.0
        # Inv: 0.5, 0.333, 0.333 -> Sum ~ 1.166
        # Norm: 0.5/1.166, 0.333/1.166...
        res = calculate_implied_probs(2.0, 3.0, 3.0)
        self.assertIsNotNone(res)
        self.assertTrue(np.isclose(sum(res), 1.0))
        self.assertTrue(res[0] < 0.5) # Normalized down due to margin

    def test_calculate_implied_probs_invalid(self):
        # One invalid odd
        self.assertIsNone(calculate_implied_probs(2.0, 1.0, 3.0)) # 1.0 invalid
        self.assertIsNone(calculate_implied_probs(2.0, None, 3.0))
        self.assertIsNone(calculate_implied_probs(2.0, -5, 3.0))

    def test_gating_logic_invariant(self):
        # Mock logic similar to backtest/predict
        min_edge = 0.05
        min_ev = 0.0
        
        # Case 1: High edge, negative EV
        # Edge = 0.1 (Passes), EV = -0.04 (Fails min_ev=0.0)
        bet = {"edge": 0.1, "ev": -0.04, "odds": 1.6}
        should_bet = (bet["edge"] >= min_edge) and (bet["ev"] >= min_ev)
        self.assertFalse(should_bet)
        
        # Case 2: Good edge, Good EV
        bet2 = {"edge": 0.1, "ev": 0.08, "odds": 1.8}
        should_bet2 = (bet2["edge"] >= min_edge) and (bet2["ev"] >= min_ev)
        self.assertTrue(should_bet2)

if __name__ == '__main__':
    unittest.main()
