import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path

def calculate_business_cost(y_true, y_probs, threshold):
    """Calculates total loss based on the ATLAS-X Cost Matrix."""
    preds = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    
    # Cost Matrix
    cost_fn = 2000  # We missed a fraudster
    cost_fp = 50    # We annoyed a legitimate customer
    
    total_loss = (fn * cost_fn) + (fp * cost_fp)
    return total_loss

def run_optimization():
    model_path = Path("src/models/atlass_x_xgb_v1.pkl")
    data_path = Path("data/processed/train_featured.parquet")
    
    print("🎯 Loading model and data for Cost Optimization...")
    model = joblib.load(model_path)
    df = pd.read_parquet(data_path).sample(50000) # Sample for speed

    X = df.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1)
    y = df['isFraud']

    # Get raw probabilities
    y_probs = model.predict_proba(X)[:, 1]

    # Test thresholds from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.95, 100)
    costs = [calculate_business_cost(y, y_probs, t) for t in thresholds]

    # Find the minimum
    best_threshold = thresholds[np.argmin(costs)]
    min_cost = min(costs)

    print(f"\n--- Optimization Results ---")
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"Minimum Estimated Loss: ${min_cost:,.2f}")
    
    # Calculate baseline (Default 0.5 threshold)
    default_cost = calculate_business_cost(y, y_probs, 0.5)
    savings = default_cost - min_cost
    print(f"Savings over default threshold: ${savings:,.2f}")

    # Visualizing the "U-Curve"
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, costs, color='crimson', lw=2)
    plt.axvline(best_threshold, color='black', linestyle='--', label=f'Best: {best_threshold:.2f}')
    plt.title("Financial Loss vs. Classification Threshold")
    plt.xlabel("Fraud Probability Threshold")
    plt.ylabel("Total Loss ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_optimization()  