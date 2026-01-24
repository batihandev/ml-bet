import pandas as pd
import numpy as np
from ML.src.dataset.cleaner import is_odds_feature, select_feature_columns, make_time_split, clean_training_data
from ML.src.features.derived import add_draw_closeness_features

def test_is_odds_feature():
    # True positives
    assert is_odds_feature("odd_home")
    assert is_odds_feature("max_odd_home")
    assert is_odds_feature("handicap_home")
    assert is_odds_feature("spread")
    assert is_odds_feature("home_vig")
    assert is_odds_feature("implied_prob_home")
    assert is_odds_feature("bookmaker_name")
    assert is_odds_feature("margin") # "margin" keyword is allowed if tokenized
    assert is_odds_feature("col_with_margin") # Split check
    
    # True negatives
    assert not is_odds_feature("home_win_rate")
    assert not is_odds_feature("form5_home_goals")
    assert not is_odds_feature("handicapper_score") # Partial token match should fail
    assert not is_odds_feature("my_baseline_feature")
    
    print("test_is_odds_feature PASSED")

def test_select_feature_columns():
    df = pd.DataFrame(columns=[
        "odd_home", "handicap_home", "home_form5_win_rate", 
        "away_form5_win_rate", "random_col", "h2h_matches_count"
    ])
    
    # 1. Check if excludes odds correctly
    cols_no_odds = select_feature_columns(df, exclude_odds=True)
    assert "odd_home" not in cols_no_odds
    assert "handicap_home" not in cols_no_odds
    assert "home_form5_win_rate" in cols_no_odds
    assert "h2h_matches_count" in cols_no_odds
    
    # 2. Check if includes odds when False
    cols_with_odds = select_feature_columns(df, exclude_odds=False)
    assert "odd_home" in cols_with_odds
    assert "handicap_home" in cols_with_odds
    
    print("test_select_feature_columns PASSED")

def test_add_draw_closeness_features():
    df = pd.DataFrame({
        "match_id": [1],
        "home_attack_strength": [1.2],
        "away_attack_strength": [0.8],
        "home_form5_goals_for_avg": [2.0],
        "away_form5_goals_for_avg": [1.0],
        "home_form5_ft_win_rate": [0.6],
        "home_form5_ft_draw_rate": [0.2],
        "away_form5_ft_win_rate": [0.3],
        "away_form5_ft_draw_rate": [0.3],
        "home_random_feature": [10],
        "away_random_feature": [5]
    })
    
    df_out = add_draw_closeness_features(df)
    
    # Check attack strength gap
    assert "gap_attack_strength" in df_out.columns
    assert df_out["gap_attack_strength"].iloc[0] == 1.2 - 0.8
    
    # Check form gap derived
    assert "gap_form5_goals_for_avg" in df_out.columns
    
    # Check PPG derived
    # home_ppg = 3*0.6 + 1*0.2 = 2.0
    # away_ppg = 3*0.3 + 1*0.3 = 1.2
    # gap = 0.8
    assert "gap_form5_ppg" in df_out.columns
    assert np.isclose(df_out["gap_form5_ppg"].iloc[0], 0.8)
    
    # Check whitelist (random feature should be ignored)
    assert "gap_random_feature" not in df_out.columns
    
    print("test_add_draw_closeness_features PASSED")

def test_make_time_split():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({"match_date": dates})
    
    # Default: ~6 months from max (max is 2023-04-10)
    # 6 months back from April is roughly Oct previous year.
    # Uh wait, this is small data. max date is 2023-04-10. -6 months is 2022-10.
    # So ALL data should be > cutoff (val_mask=True), none in train?
    # correct.
    train, val = make_time_split(df)
    assert val.all() # All data is within "last 6 months"
    
    # Fixed cutoff
    train_fix, val_fix = make_time_split(df, fixed_cutoff="2023-02-01")
    assert train_fix.sum() == 31 # 31 days in Jan
    
    print("test_make_time_split PASSED")

def test_clean_training_data():
    df = pd.DataFrame({
        "target": [0, 1, 0, 1, 0],
        "col1": [1, 1, 1, 1, 1],
        "col2": [1, 1, 1, 1, np.nan], # missing last one
        "match_date": pd.date_range("2023-01-01", periods=5)
    })
    # Mock feature cols
    feats = ["col1", "col2"]
    # col2 is rolling? No "form" in name, so history filter won't apply to it unless we rename
    
    # Rename for testing history
    df = df.rename(columns={"col2": "home_form5_val"})
    feats = ["col1", "home_form5_val"]
    
    X_tr, X_val, y_tr, y_val, f_cols = clean_training_data(df, feats, "target", fixed_cutoff="2023-01-03")
    
    # Should accept row 4 (index 4 is 5th) despite NaN?
    # threshold: max(1, 0.2*1) = 1.
    # last row has 0 valid rolling cols (since col2 is NaN). So it should be dropped?
    # Wait, rolling_cols=["home_form5_val"]. len=1.
    # Row 4: home_form5_val is NaN. sum=0. 0 < 1. 
    # Yes, should be dropped.

    # Total rows: 5. Row 4 dropped -> 4 rows left.
    # Split "2023-01-03":
    # 01-01 (train), 01-02 (train), 01-03 (val), 01-04 (val).
    # so 2 train, 2 val.
    
    assert len(X_tr) + len(X_val) == 4
    
    print("test_clean_training_data PASSED")

if __name__ == "__main__":
    test_is_odds_feature()
    test_select_feature_columns()
    test_add_draw_closeness_features()
    test_make_time_split()
    test_clean_training_data()
    print("ALL TESTS PASSED")
