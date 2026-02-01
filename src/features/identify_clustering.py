import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def build_identity_clusters():
    input_file = Path("data/processed/train_featured.parquet")
    output_file = Path("data/processed/train_clustered.parquet")
    
    print("🧬 Starting Identity Link Analysis (K-Means)...")
    df = pd.read_parquet(input_file)

    # 1. Select Identity-Based Features
    # We want to group by "Who" is doing the transaction
    id_features = [
        'card1', 'card2', 'card3', 'card5', 
        'addr1', 'TransactionAmt_Log', 'Transaction_Hour'
    ]
    
    # Handle missing values specifically for clustering
    cluster_data = df[id_features].fillna(-1)

    # 2. Scaling (Essential for K-Means)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_data)

    # 3. K-Means Clustering
    # We use k=15 to identify broad "User Persona" groups
    print("--- Fitting K-Means on Identity Signatures ---")
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    df['identity_cluster'] = kmeans.fit_transform(scaled_features).argmin(axis=1)

    # 4. Calculate Cluster Risk (The "Value Add")
    # This is a supervised feature derived from unsupervised grouping
    print("--- Calculating Cluster Risk Scores ---")
    cluster_risk = df.groupby('identity_cluster')['isFraud'].mean().to_dict()
    df['cluster_fraud_rate'] = df['identity_cluster'].map(cluster_risk)

    # 5. Save Results
    df.to_parquet(output_file, engine='pyarrow')
    print(f"✅ Identity Resolution Complete! New Feature: 'cluster_fraud_rate'")
    print(f"Sample Risk Rates: {list(cluster_risk.values())[:5]}")

if __name__ == "__main__":
    build_identity_clusters()