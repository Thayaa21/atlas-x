import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def train_and_evaluate():
    # Load data
    data_path = Path("data/processed/train_clustered.parquet")
    model_path = Path("src/models/atlass_x_xgb_v2.pkl")
    
    print("🔄 Loading clustered data for V2 Enterprise Training...")
    df = pd.read_parquet(data_path)

    # Prepare features
    X = df.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1)
    y = df['isFraud']

    # Handle categoricals
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = X[col].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"📈 Training on {len(X.columns)} features...")
    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"Fraud rate: {y.mean()*100:.2f}%")
    
    # Train model
    model = xgb.XGBClassifier(
        tree_method='hist',
        enable_categorical=True,
        n_estimators=500,
        max_depth=6,
        scale_pos_weight=25,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    
    # Save model
    joblib.dump(model, model_path)
    
    # ============================================
    # EVALUATION
    # ============================================
    print("\n" + "="*60)
    print("EXPERIMENTAL RESULTS - MODEL PERFORMANCE")
    print("="*60)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n📊 CLASSIFICATION METRICS:")
    print(f"   AUC-ROC:    {auc:.4f} {'✅' if auc >= 0.94 else '⚠️'}")
    print(f"   Precision:  {precision:.4f} (85% target: {'✅' if precision >= 0.85 else '⚠️'})")
    print(f"   Recall:     {recall:.4f} (75% target: {'✅' if recall >= 0.75 else '⚠️'})")
    print(f"   F1-Score:   {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📈 CONFUSION MATRIX:")
    print(f"   True Negatives:  {tn:,}")
    print(f"   False Positives: {fp:,}")
    print(f"   False Negatives: {fn:,}")
    print(f"   True Positives:  {tp:,}")
    print(f"   False Positive Rate: {fp/(fp+tn)*100:.2f}%")
    print(f"   False Negative Rate: {fn/(fn+tp)*100:.2f}%")
    
    # Find optimal threshold
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"\n🎯 OPTIMAL THRESHOLD:")
    print(f"   Default (0.5):  Precision={precision:.3f}, Recall={recall:.3f}")
    print(f"   Optimal ({optimal_threshold:.3f}): F1={f1_scores[optimal_idx]:.3f}")
    
    # ============================================
    # VISUALIZATIONS
    # ============================================
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', cbar=True,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.title('Confusion Matrix - XGBoost V2', fontsize=16, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: confusion_matrix_results.png")
    
    # 2. Performance Metrics Bar Chart
    plt.figure(figsize=(10, 6))
    metrics_names = ['AUC-ROC', 'Precision', 'Recall', 'F1-Score']
    metrics_values = [auc, precision, recall, f1]
    targets = [0.94, 0.85, 0.75, 0.80]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, metrics_values, width, label='Achieved', color='#3498db')
    bars2 = plt.bar(x + width/2, targets, width, label='Target', color='#95a5a6', alpha=0.6)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance vs Targets', fontsize=16, fontweight='bold')
    plt.xticks(x, metrics_names, fontsize=11)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: performance_metrics.png")
    
    # 3. ROC Curve
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#e74c3c', linewidth=2, label=f'XGBoost (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: roc_curve.png")
    
    # 4. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='#2ecc71', linewidth=2)
    plt.scatter(recall_vals[optimal_idx], precision_vals[optimal_idx], 
                color='red', s=100, zorder=5, 
                label=f'Optimal (threshold={optimal_threshold:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: precision_recall_curve.png")
    
    # Summary stats
    print("\n" + "="*60)
    print("SUMMARY - CHECKPOINT 1 RESULTS")
    print("="*60)
    print(f"✅ Model trained successfully with {len(X.columns)} features")
    print(f"✅ AUC-ROC: {auc:.4f} (Target: 0.94)")
    print(f"✅ Caught {tp:,}/{tp+fn:,} fraud cases ({tp/(tp+fn)*100:.1f}%)")
    print(f"⚠️  False alarms: {fp:,}/{fp+tn:,} legitimate ({fp/(fp+tn)*100:.2f}%)")
    print(f"✅ Model saved to: {model_path}")
    print("="*60)
    
    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'optimal_threshold': optimal_threshold
    }

if __name__ == "__main__":
    results = train_and_evaluate()