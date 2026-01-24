import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from dataset.cleaner import clean_training_data, select_feature_columns
from .schema import TARGET_COL, CLASS_MAPPING

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
    train_start: str = None,
    train_end: str = None,
    cutoff_date: str = None,
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
    
    # Filter by date if provided
    if train_start:
        df = df[df["match_date"] >= pd.to_datetime(train_start)]
    if train_end:
        df = df[df["match_date"] <= pd.to_datetime(train_end)]
        
    # Prepare target
    df = prepare_1x2_target(df)
    
    # Select features
    feature_cols = select_feature_columns(df)
    
    # Clean and split
    X_train, X_val, y_train, y_val, used_features = clean_training_data(df, feature_cols, TARGET_COL, cutoff_date)
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
    print(f"Target distribution (train): {pd.Series(y_train).value_counts(normalize=True).to_dict()}")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    report_dict = classification_report(y_val, val_pred, output_dict=True)
    report_str = classification_report(y_val, val_pred)
    
    print(f"Validation Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", report_str)
    
    # Feature importance
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:30]
    feature_importance = [
        {"feature": used_features[i], "importance": float(importances[i])}
        for i in indices
    ]
    
    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "model_ft_1x2.joblib"
    joblib.dump({
        "model": model,
        "features": used_features,
        "target": TARGET_COL,
        "feature_importance": feature_importance,
        "class_mapping": CLASS_MAPPING
    }, model_path)
    
    # Save meta
    with open(MODELS_DIR / "model_ft_1x2_meta.json", "w") as f:
        json.dump({
            "target": TARGET_COL,
            "model_file": "model_ft_1x2.joblib",
            "n_features": len(used_features),
            "feature_importance": feature_importance,
            "training_params": {
                "train_start": train_start,
                "train_end": train_end,
                "cutoff_date": str(cutoff_date) if cutoff_date else None,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf
            },
            "metrics": {
                "accuracy": acc,
                "classification_report": report_dict,
                "n_train": len(X_train),
                "n_val": len(X_val)
            }
        }, f, indent=2)
        
    print(f"Production model saved to {model_path}")

if __name__ == "__main__":
    train_production_model()
