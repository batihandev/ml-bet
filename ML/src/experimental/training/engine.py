import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from .data import clean_training_data

ROOT_DIR = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT_DIR / "models"

def train_single_model(
    df, feature_cols, target_col, model_name, 
    fixed_cutoff=None,
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=50
):
    print(f"\n=== Training model for target: {target_col} ===")
    X_train, X_val, y_train, y_val, used_features = clean_training_data(df, feature_cols, target_col, fixed_cutoff)

    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1, random_state=42, class_weight="balanced",
    )
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_val, val_proba)
        ap = average_precision_score(y_val, val_proba)
        print(f"AUC: {auc:.4f}, Average Precision: {ap:.4f}")
    except:
        pass

    # Calculate feature importance
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:20]  # Top 20
    feature_importance = [
        {"feature": used_features[i], "importance": float(importances[i])}
        for i in indices
    ]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump({
        "model": model, 
        "features": used_features, 
        "target": target_col,
        "feature_importance": feature_importance
    }, model_path)

    with open(MODELS_DIR / f"{model_name}_meta.json", "w") as f:
        json.dump({
            "target": target_col, 
            "model_file": str(model_path), 
            "n_features": len(used_features),
            "feature_importance": feature_importance
        }, f, indent=2)
    
    print(f"Saved model to {model_path}")
