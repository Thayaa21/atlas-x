import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_txn = pd.read_csv('data/raw/train_transaction.csv')
train_id = pd.read_csv('data/raw/train_identity.csv')
df = train_txn.merge(train_id, on='TransactionID', how='left')

fraud = df[df['isFraud'] == 1]
legit = df[df['isFraud'] == 0]

print("="*50)
print("GENERATING IMPROVED KEY INSIGHTS CHARTS")
print("="*50)

# ============================================
# CHART 1: Missing Values (SIMPLIFIED)
# ============================================
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
high_missing = missing_pct[missing_pct > 40]

# Create simple bar chart instead of heatmap
plt.figure(figsize=(12, 6))
top_15_missing = missing_pct.head(15)

bars = plt.barh(range(len(top_15_missing)), top_15_missing.values)

# Color code by severity
colors = ['#c0392b' if x > 90 else '#e74c3c' if x > 70 else '#e67e22' if x > 50 else '#f39c12' 
          for x in top_15_missing.values]
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.yticks(range(len(top_15_missing)), top_15_missing.index, fontsize=11)
plt.xlabel('Missing Percentage (%)', fontsize=13, fontweight='bold')
plt.title(f'Top 15 Features with Missing Values\n({len(high_missing)} total columns have >40% missing)', 
          fontsize=15, fontweight='bold')

# Add percentage labels
for i, (idx, val) in enumerate(top_15_missing.items()):
    plt.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

plt.xlim(0, 105)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('missing_values_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Chart 1 saved: {len(high_missing)} columns with >40% missing")

# ============================================
# CHART 2: Correlation (TOP 10 ONLY - CLEANER)
# ============================================
# Select fewer features for readability
key_features = ['TransactionAmt', 'card1', 'card2', 'card5', 
                'C1', 'C2', 'C13', 'C14', 'D1', 'D2']
key_features = [f for f in key_features if f in df.columns]

corr_matrix = df[key_features].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # Hide upper triangle

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='RdYlGn_r', center=0, square=True, linewidths=1,
            cbar_kws={'label': 'Correlation', 'shrink': 0.8},
            vmin=-1, vmax=1)

plt.title('Feature Correlation Matrix\n(Key Features Only)', fontsize=15, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 2 saved: Simplified correlation matrix")

# ============================================
# CHART 3: Velocity Pattern (FIXED CALCULATION)
# ============================================
# Better velocity calculation: transactions per card
fraud_cards = fraud.groupby('card1').size()
legit_cards = legit.groupby('card1').size()

fraud_avg = fraud_cards.mean()
legit_avg = legit_cards.mean()
velocity_ratio = fraud_avg / legit_avg

print(f"\nVelocity Stats:")
print(f"  Fraud avg: {fraud_avg:.2f} txns/card")
print(f"  Legit avg: {legit_avg:.2f} txns/card")
print(f"  Ratio: {velocity_ratio:.2f}x")

plt.figure(figsize=(10, 7))
categories = ['Legitimate\nCards', 'Fraudulent\nCards']
values = [legit_avg, fraud_avg]
colors = ['#27ae60', '#e74c3c']

bars = plt.bar(categories, values, color=colors, edgecolor='black', linewidth=2, width=0.6)

# Add value labels
for bar, val in zip(bars, values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{val:.1f}\ntransactions\nper card',
             ha='center', va='bottom', fontsize=13, fontweight='bold')


plt.text(0.5, max(values) * 0.55, 
         f'Legitimate cards\nshow {1/velocity_ratio:.1f}x higher\ntransaction volume\n(fraudsters burn through cards faster)',
         ha='center', fontsize=13, fontweight='bold',
         bbox=dict(boxstyle='round,pad=1', facecolor='#fff3cd', 
                   edgecolor='#ff6b6b', linewidth=3))

plt.ylabel('Average Transactions per Card', fontsize=13, fontweight='bold')
plt.title('Transaction Volume: Fraud vs Legitimate Cards', fontsize=16, fontweight='bold')
plt.ylim(0, max(values) * 1.3)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('velocity_pattern.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Chart 3 saved: {velocity_ratio:.1f}x velocity difference")

if 'DeviceInfo' in df.columns:
    device_cards = df.groupby('DeviceInfo')['card1'].nunique().reset_index()
    device_cards.columns = ['DeviceInfo', 'num_cards']
    
    # Categorize - FIXED: 7 bins need 6 labels
    device_cards['category'] = pd.cut(device_cards['num_cards'], 
                                      bins=[0, 2, 3, 4, 5, 100, 5000],  # 6 edges
                                      labels=['1 card', '2 cards', '3 cards', 
                                             '4 cards', '5-100 cards', '100+ cards'],  # 6 labels
                                      include_lowest=True)
    
    category_counts = device_cards['category'].value_counts().sort_index()
    
    # Get max for title
    max_cards = device_cards['num_cards'].max()
    ghost_count = len(device_cards[device_cards['num_cards'] >= 5])
    
    print(f"\nGhost Persona Stats:")
    print(f"  Ghost devices (5+ cards): {ghost_count}")
    print(f"  Worst device: {max_cards} cards")
    
    # Create clean bar chart
    plt.figure(figsize=(12, 7))
    
    colors_list = ['#27ae60', '#3498db', '#f39c12', '#e67e22', '#e74c3c', '#8e44ad']
    bars = plt.bar(range(len(category_counts)), category_counts.values, 
                   color=colors_list[:len(category_counts)], 
                   edgecolor='black', linewidth=1.5)
    
    plt.xticks(range(len(category_counts)), category_counts.index, fontsize=12, fontweight='bold')
    plt.xlabel('Cards per Device', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Devices', fontsize=13, fontweight='bold')
    plt.title(f'Ghost Persona Detection: Device-Card Relationships\nWorst Case: 1 device used by {max_cards} different cards', 
              fontsize=15, fontweight='bold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, category_counts.values)):
        plt.text(i, val + max(category_counts.values)*0.02, 
                f'{int(val):,}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Highlight ghost zone (after index 3 = after "4 cards")
    plt.axvline(x=3.5, color='red', linestyle='--', linewidth=3, alpha=0.7)
    plt.text(4.5, max(category_counts.values) * 0.8, 
             f'← GHOST DEVICES\n{ghost_count:,} devices\nused by 5+ cards',
             fontsize=13, fontweight='bold', color='red',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#fff3cd', 
                      edgecolor='red', linewidth=2))
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('ghost_persona.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Chart 4 saved: {ghost_count} ghost devices detected")

else:
     print("⚠ DeviceInfo not available")