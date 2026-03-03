import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load data
train_txn = pd.read_csv('data/raw/train_transaction.csv')
train_id = pd.read_csv('data/raw/train_identity.csv')

# Merge
df = train_txn.merge(train_id, on='TransactionID', how='left')

# Separate fraud vs legit
fraud = df[df['isFraud'] == 1]
legit = df[df['isFraud'] == 0]

print("Data loaded successfully!")
print(f"Total transactions: {len(df)}")
print(f"Fraud: {len(fraud)} ({len(fraud)/len(df)*100:.2f}%)")
print(f"Legit: {len(legit)} ({len(legit)/len(df)*100:.2f}%)")

# ============================================
# CHART 1: Class Distribution
# ============================================
plt.figure(figsize=(8, 6))
fraud_pct = df['isFraud'].value_counts(normalize=True) * 100
colors = ['#2ecc71', '#e74c3c']
plt.pie(fraud_pct, labels=['Legit (96.5%)', 'Fraud (3.5%)'], 
        autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Class Distribution - Severe Imbalance', fontsize=16, fontweight='bold')
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 1 saved: class_distribution.png")

# ============================================
# CHART 2: Feature Importance
# ============================================
# Quick feature selection (remove IDs, target)
X = df.drop(['TransactionID', 'TransactionDT', 'isFraud'], axis=1)
y = df['isFraud']

# Fill NaN with -999 (quick)
X = X.fillna(-999)

# Select only numeric
X = X.select_dtypes(include=[np.number])

# Train quick model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
print("Training XGBoost model...")
model.fit(X_train, y_train)

# Get importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

# Add readable names
importance['display_name'] = importance['feature'].apply(
    lambda x: f"{x} (Vesta)" if x.startswith('V') else
              f"{x} (Count)" if x.startswith('C') else
              f"{x} (Time)" if x.startswith('D') else
              f"{x} (Identity)" if x.startswith('id_') else
              "Transaction Amount" if x == 'TransactionAmt' else
              f"{x} (Card)" if x.startswith('card') else x
)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance['display_name'], importance['importance'], color='#3498db')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 10 Most Important Features', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 2 saved: feature_importance.png")

# ============================================
# CHART 3: Fraud by Hour
# ============================================
# Extract hour
df['hour'] = (df['TransactionDT'] % 86400) // 3600

# Fraud rate by hour
fraud_by_hour = df.groupby('hour')['isFraud'].mean() * 100

plt.figure(figsize=(12, 6))
plt.plot(fraud_by_hour.index, fraud_by_hour.values, 
         marker='o', linewidth=2, color='#e74c3c', markersize=8)
plt.axhline(y=3.5, color='gray', linestyle='--', label='Overall Rate (3.5%)')
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Fraud Rate (%)', fontsize=12)
plt.title('Fraud Rate by Hour - Spike at Night (2-6 AM)', fontsize=16, fontweight='bold')
plt.xticks(range(0, 24))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('fraud_by_hour.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 3 saved: fraud_by_hour.png")
print(f"  Night fraud rate (2-6 AM): {fraud_by_hour[2:7].mean():.2f}%")
print(f"  Day fraud rate (10-18): {fraud_by_hour[10:19].mean():.2f}%")

# ============================================
# CHART 4: High-Risk Regions
# ============================================
# Use addr1 as proxy for country/region
fraud_by_country = df.groupby('addr1').agg({
    'isFraud': ['sum', 'count', 'mean']
}).reset_index()
fraud_by_country.columns = ['addr1', 'fraud_count', 'total', 'fraud_rate']

# Filter: at least 1000 transactions
fraud_by_country = fraud_by_country[fraud_by_country['total'] >= 1000]
top10 = fraud_by_country.nlargest(10, 'fraud_rate')

# Create labels with transaction counts
top10['label'] = top10.apply(
    lambda row: f"Region {int(row['addr1'])} ({int(row['total'])} txns)", 
    axis=1
)

plt.figure(figsize=(11, 6))
bars = plt.barh(top10['label'], top10['fraud_rate'] * 100, color='#e67e22')

# Add percentage labels on bars
for i, (bar, rate) in enumerate(zip(bars, top10['fraud_rate'] * 100)):
    plt.text(rate + 0.15, i, f'{rate:.1f}%', 
             va='center', fontsize=10, fontweight='bold')

plt.xlabel('Fraud Rate (%)', fontsize=12)
plt.ylabel('Billing Region', fontsize=12)
plt.title('Top 10 High-Risk Billing Regions', fontsize=16, fontweight='bold')
plt.axvline(x=3.5, color='gray', linestyle='--', linewidth=1, 
            label='Overall avg (3.5%)', alpha=0.7)
plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('fraud_by_region.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 4 saved: fraud_by_region.png")

# ============================================
# CHART 5: Transaction Amount Distribution
# ============================================
plt.figure(figsize=(12, 6))

# Sample to avoid plotting 590K points
fraud_sample = fraud['TransactionAmt'].sample(min(10000, len(fraud)), random_state=42)
legit_sample = legit['TransactionAmt'].sample(min(10000, len(legit)), random_state=42)

# Violin plot
data_to_plot = [legit_sample, fraud_sample]
parts = plt.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)

# Color the violins
for pc in parts['bodies']:
    pc.set_facecolor('#3498db')
    pc.set_alpha(0.7)

plt.xticks([1, 2], ['Legit', 'Fraud'], fontsize=12)
plt.ylabel('Transaction Amount ($)', fontsize=12)
plt.title('Transaction Amount Distribution - Fraud vs Legit', fontsize=16, fontweight='bold')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('amount_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 5 saved: amount_distribution.png")
print(f"  Fraud median: ${fraud['TransactionAmt'].median():.2f}")
print(f"  Legit median: ${legit['TransactionAmt'].median():.2f}")

print("\n" + "="*50)
print("ALL CHARTS GENERATED SUCCESSFULLY!")
print("="*50)
print("Files created:")
print("  1. class_distribution.png")
print("  2. feature_importance.png")
print("  3. fraud_by_hour.png")
print("  4. fraud_by_region.png")
print("  5. amount_distribution.png")