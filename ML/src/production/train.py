import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import classification_report, accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from dataset.cleaner import clean_training_data, select_feature_columns
from .schema import TARGET_COL, CLASS_MAPPING
from .utils import build_time_folds, ProbabilityCalibrator

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

def prepare_1x2_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a multiclass target column from FT results."""
    df = df.copy()
    
    # We can use ft_home_goals vs ft_away_goals
    def get_outcome(row):
        h = row.get("ft_home_goals")
        a = row.get("ft_away_goals")
        if pd.isna(h) or pd.isna(a):
            return np.nan
        if h > a: return CLASS_MAPPING["home"]
        if h < a: return CLASS_MAPPING["away"]
        return CLASS_MAPPING["draw"]

    df[TARGET_COL] = df.apply(get_outcome, axis=1)
    return df

def train_production_model(
    train_start: str = "2020-06-30",
    training_cutoff_date: str = "2025-06-30",
    oof_calibration: bool = True,
    calibration_method: str = "none",
    oof_step: str = "1 month",
    oof_min_train_span: str = "24 months",
    n_estimators: int = 300,
    max_depth: int = 8,
    min_samples_leaf: int = 50
):
    print("\n=== Training Production 1X2 Model ===")
    
    # Load features
    feat_path = PROCESSED_DIR / "features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    
    df = pd.read_csv(feat_path, parse_dates=["match_date"], low_memory=False)
    
    # Filter rows by date if provided (Initial filter)
    if train_start:
        df = df[df["match_date"] >= pd.to_datetime(train_start)]
    if training_cutoff_date:
        df = df[df["match_date"] <= pd.to_datetime(training_cutoff_date)]
        
    # Prepare target
    df = prepare_1x2_target(df)
    
    # Select features (EXCLUDING ODDS)
    print("Selecting features (excluding odds)...")
    feature_cols = select_feature_columns(df, exclude_odds=True)
    
    # Filter rows where target is missing
    df = df[~df[TARGET_COL].isna()].copy()
    
    # Enforce date windows
    t_start_dt = pd.to_datetime(train_start) if train_start else df["match_date"].min()
    # Default to data end if cutoff is not provided
    t_cutoff_dt = pd.to_datetime(training_cutoff_date) if training_cutoff_date else pd.to_datetime("2025-06-30")
    
    # 1. OOF Calibration Loop
    oof_preds = []
    oof_targets = []
    oof_dates = []
    
    # If OOF disabled, force calibration to none
    if not oof_calibration and calibration_method != "none":
        print(f"WARNING: OOF calibration loop disabled but method='{calibration_method}' requested. Falling back to 'none'.")
        calibration_method = "none"
    
    if oof_calibration:
        print(f"Generating OOF predictions: {t_start_dt.date() if hasattr(t_start_dt, 'date') else t_start_dt} to {t_cutoff_dt.date() if hasattr(t_cutoff_dt, 'date') else t_cutoff_dt}")
        # Pass min_fold_rows to avoid tiny folds at the end
        folds = build_time_folds(df, train_start, training_cutoff_date or "2025-06-30", step=oof_step, min_train_span=oof_min_train_span, min_fold_rows=100)
        print(f"Number of OOF folds: {len(folds)}")
        
        for i, (train_f, test_f, f_start, f_end) in enumerate(folds):
            print(f"  Fold {i+1}/{len(folds)}: Test [{f_start.date()} to {f_end.date()}] (Train samples: {len(train_f)})")
            
            X_f_train = train_f[feature_cols].fillna(0.0)
            y_f_train = train_f[TARGET_COL].astype(int).values
            X_f_test = test_f[feature_cols].fillna(0.0)
            y_f_test = test_f[TARGET_COL].astype(int).values
            
            fold_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced"
            )
            fold_model.fit(X_f_train, y_f_train)
            
            # 3. Confirm Column Alignment
            # print(f"    Fold classes: {fold_model.classes_}") # Log spam check, keeping minimal
            
            p_fold_raw = fold_model.predict_proba(X_f_test)
            
            # 1. Expand p_fold_raw to (N, 3) to handle missing classes in this fold
            n_test = len(X_f_test)
            p_full = np.zeros((n_test, 3))
            
            # map class indices from fold_model to 0,1,2
            for idx, cls in enumerate(fold_model.classes_):
                if 0 <= cls < 3:
                    p_full[:, int(cls)] = p_fold_raw[:, idx]
            
            oof_preds.append(p_full)
            oof_targets.append(y_f_test)
            oof_dates.extend(test_f["match_date"].tolist())

        if not oof_preds:
            print("WARNING: No OOF folds generated. Check training_cutoff_date and min_train_span.")
            calibrator = ProbabilityCalibrator(method="none")
            oof_metrics = {}
        else:
            oof_preds_agg = np.vstack(oof_preds)
            oof_targets_agg = np.concatenate(oof_targets)
            
            # Fit Calibrator on OOF data
            print(f"Fitting ProbabilityCalibrator (method='{calibration_method}')...")
            calibrator = ProbabilityCalibrator(method=calibration_method)
            calibrator.fit(oof_preds_agg, oof_targets_agg)
            
            # Evaluate OOF
            p_cal = calibrator.transform(oof_preds_agg)
            y_pred = np.argmax(p_cal, axis=1)
            
            acc = accuracy_score(oof_targets_agg, y_pred)
            from sklearn.metrics import log_loss
            loss = log_loss(oof_targets_agg, p_cal)
            
            # Multiclass Brier
            y_true_onehot = pd.get_dummies(oof_targets_agg).reindex(columns=[0, 1, 2], fill_value=0).values
            brier = np.mean(np.sum((p_cal - y_true_onehot)**2, axis=1))
            
            report_dict = classification_report(oof_targets_agg, y_pred, output_dict=True)
            report_str = classification_report(oof_targets_agg, y_pred)
            
            # --- 5 & 6. Draw Specific Analysis ---
            draw_idx = CLASS_MAPPING["draw"]
            
            # Raw predictions (argmax)
            y_pred_raw = np.argmax(oof_preds_agg, axis=1)
            
            # Draw Recall/F1 Comparison
            from sklearn.metrics import recall_score, f1_score
            rec_raw = recall_score(oof_targets_agg, y_pred_raw, labels=[draw_idx], average=None)[0]
            f1_raw = f1_score(oof_targets_agg, y_pred_raw, labels=[draw_idx], average=None)[0]
            
            rec_cal = recall_score(oof_targets_agg, y_pred, labels=[draw_idx], average=None)[0]
            f1_cal = f1_score(oof_targets_agg, y_pred, labels=[draw_idx], average=None)[0]
            
            print("\n=== Draw Class Analysis ===")
            print(f"Draw Recall: Raw={rec_raw:.4f} -> Calibrated={rec_cal:.4f}")
            print(f"Draw F1    : Raw={f1_raw:.4f} -> Calibrated={f1_cal:.4f}")
            
            # Mean Probabilities
            is_draw = (oof_targets_agg == draw_idx)
            mean_p_draw_true = p_cal[is_draw, draw_idx].mean()
            mean_p_draw_false = p_cal[~is_draw, draw_idx].mean()
            print(f"Mean P(Draw) | True Draw: {mean_p_draw_true:.4f}")
            print(f"Mean P(Draw) | Not Draw : {mean_p_draw_false:.4f}")
            
            # --- 7. Betting Metrics (Simple) ---
            # Assume constant odds for quick check or join with odds if available
            # Note: We don't have odds in 'oof_preds' / 'oof_targets_agg' directly unless we kept them.
            # We have 'oof_dates' and indices, but joining back is complex here.
            # Instead, we will rely on generic Expected Value proxy or just skip ROI for this run if odds aren't handy.
            # Actually, let's skip ROI here to avoid complexity of re-fetching odds, focusing on prob stats.
            print("===========================\n")
            
            
            oof_metrics = {
                "accuracy": acc,
                "log_loss": loss,
                "brier_score": brier,
                "classification_report": report_dict,
                "n_folds": len(folds),
                "n_oof_rows": len(oof_targets_agg),
                "oof_date_range": f"{min(oof_dates).date()} to {max(oof_dates).date()}"
            }
            
            print(f"OOF Accuracy: {acc:.4f}")
            print(f"OOF Log Loss: {loss:.4f}")
            print(f"OOF Brier Score: {brier:.4f}")
            print("\nClassification Report (OOF):\n", report_str)
    else:
        print("OOF Calibration DISABLED. Training final model only.")
        calibrator = None
        oof_metrics = {}

    # 2. Train Final Base Model on ALL data
    final_df = df[(df["match_date"] >= t_start_dt) & (df["match_date"] <= t_cutoff_dt)].copy()
    X_final = final_df[feature_cols].fillna(0.0)
    y_final = final_df[TARGET_COL].astype(int).values
    
    print(f"Training final model on {len(X_final)} samples ({t_start_dt.date()} to {t_cutoff_dt.date()})")
    
    base_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"
    )
    base_model.fit(X_final, y_final)
    
    # Feature importance from base model
    importances = base_model.feature_importances_
    indices = importances.argsort()[::-1][:30]
    feature_importance = [
        {"feature": feature_cols[i], "importance": float(importances[i])}
        for i in indices
    ]
    
    # Dynamic data end
    real_data_end = str(df["match_date"].max().date())

    # Save model bundle
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "model_ft_1x2.joblib"
    joblib.dump({
        "base_model": base_model,
        "calibrator": calibrator,
        "features": feature_cols,
        "class_mapping": CLASS_MAPPING,
        "windows": {
            "train_start": str(t_start_dt.date()),
            "training_cutoff_date": str(t_cutoff_dt.date()),
            "data_end": real_data_end
        },
        "oof": oof_metrics,
        "feature_importance": feature_importance
    }, model_path)
    
    # Save meta
    with open(MODELS_DIR / "model_ft_1x2_meta.json", "w") as f:
        json.dump({
            "target": TARGET_COL,
            "model_file": "model_ft_1x2.joblib",
            "n_features": len(feature_cols),
            "feature_importance": feature_importance,
            "training_params": {
                "train_start": train_start,
                "training_cutoff_date": training_cutoff_date,
                "oof_calibration": oof_calibration,
                "oof_step": oof_step,
                "oof_min_train_span": oof_min_train_span,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf
            },
            "metrics": oof_metrics,
            "backtest_eligible_from": str((t_cutoff_dt + pd.Timedelta(days=1)).date()),
            "data_end": real_data_end
        }, f, indent=2)
        
    print(f"Production model saved to {model_path}")

if __name__ == "__main__":
    train_production_model()
