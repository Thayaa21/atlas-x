import pandas as pd
import numpy as np
from pathlib import Path

def build_features():
    processed_path = Path("data/processed")
    input_file = processed_path / "train_merged.parquet"
    output_file = processed_path / "train_featured.parquet"

    print("🛠️  Starting Feature Engineering...")
    df = pd.read_parquet(input_file)

    # 1. Time-Cycle Engineering
    # The dataset starts at a specific (unnamed) point in time. 
    # TransactionDT is an offset in seconds.
    print("--- Engineering Time Features ---")
    df['Transaction_Hour'] = np.floor((df['TransactionDT'] / 3600) % 24)
    df['Transaction_Day'] = np.floor((df['TransactionDT'] / (3600 * 24)) % 7)

    # 2. Card Velocity & Identity Mapping
    # 'card1'-'card6' and 'addr1' are the primary identifiers for a "user"
    print("--- Engineering Velocity Features ---")
    
    # Create a unique ID for a 'card'/account
    df['uid'] = df['card1'].astype(str) + '_' + df['card2'].astype(str) + '_' + df['card3'].astype(str)

    # Transaction Count per UID (Historical Velocity)
    df['uid_count'] = df.groupby(['uid'])['TransactionID'].transform('count')

    # Average Transaction Amount per UID
    df['uid_TransactionAmt_mean'] = df.groupby(['uid'])['TransactionAmt'].transform('mean')
    
    # Deviation from the mean (Is this transaction unusually large for this user?)
    df['uid_Amt_Relative_Check'] = df['TransactionAmt'] / df['uid_TransactionAmt_mean']

    # 3. Log Transformation of Transaction Amount
    # Financial data is often heavily skewed; log helps XGBoost converge faster
    df['TransactionAmt_Log'] = np.log1p(df['TransactionAmt'])

    # 4. Clean up temporary columns
    df.drop(['uid'], axis=1, inplace=True)

    # Save the featured dataset
    print(f"--- Saving featured data to {output_file} ---")
    df.to_parquet(output_file, engine='pyarrow')
    
    print(f"✅ Feature Engineering Complete! New Shape: {df.shape}")

if __name__ == "__main__":
    build_features()