import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path

def retrain_with_clusters():
    # Use the clustered data from Task 5
    data_path = Path("data/processed/train_clustered.parquet")
    model_path = Path("src/models/atlass_x_xgb_v2.pkl") # Note: V2!
    
    print("🔄 Loading clustered data for V2 Enterprise Training...")
    df = pd.read_parquet(data_path)

    # CRITICAL: Ensure we drop ONLY the non-predictive columns
    # 'isFraud' is the target, others are ID/Time markers
    X = df.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1)
    y = df['isFraud']

    # Handle Categoricals (Essential for M3 Mac performance)
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = X[col].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    print(f"📈 Training on {len(X.columns)} features (including Clusters)...")
    
    model = xgb.XGBClassifier(
        tree_method='hist',
        enable_categorical=True,
        n_estimators=500,
        max_depth=6,
        scale_pos_weight=25, # Adjusted for typical IEEE-CIS imbalance
        learning_rate=0.05
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    
    joblib.dump(model, model_path)
    print(f"✅ Model V2 Saved. Feature count: {len(model.feature_names_in_)}")

if __name__ == "__main__":
    retrain_with_clusters()