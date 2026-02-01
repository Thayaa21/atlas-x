import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from pathlib import Path

def train_xgboost():
    processed_path = Path("data/processed")
    model_path = Path("src/models")
    input_file = processed_path / "train_featured.parquet"

    print("🏗️  Loading featured data for training...")
    df = pd.read_parquet(input_file)

    # 1. Define Target and Features
    # Dropping non-predictive IDs and the target itself
    X = df.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1)
    y = df['isFraud']

    # Handle Categorical Columns for XGBoost
    for col in X.select_dtypes(include=['category']).columns:
        X[col] = X[col].astype('object').fillna('None').astype('category')

    # 2. Train/Test Split (Time-series aware split would be better, but we'll start with random)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Handle Imbalance: Calculate scale_pos_weight
    # scale_pos_weight = count(negative) / count(positive)
    fraud_weight = (y == 0).sum() / (y == 1).sum()

    print(f"--- Training XGBoost with Scale Weight: {fraud_weight:.2f} ---")

    # 4. Initialize and Train XGBoost
    # Using 'hist' tree_method for speed on M3 Mac
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method='hist', 
        enable_categorical=True, # Critical for memory downcasted categories
        scale_pos_weight=fraud_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              verbose=50)

    # 5. Evaluation
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation ---")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")

    # 6. Save Model
    joblib.dump(model, model_path / "atlass_x_xgb_v1.pkl")
    print(f"✅ Model saved to {model_path / 'atlass_x_xgb_v1.pkl'}")

if __name__ == "__main__":
    train_xgboost()