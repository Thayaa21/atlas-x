import pandas as pd
import numpy as np
import time
import os
from pathlib import Path

def reduce_mem_usage(df, verbose=True):
    """
    Iterates through all columns of a dataframe and modifies the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32) # float16 is often slow on CPU, using float32
                else:
                    df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({((start_mem - end_mem) / start_mem):.1%}-reduction)')
    return df

def ingest_data():
    # Setup Paths
    raw_path = Path("data/raw")
    processed_path = Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    print("🚀 Starting Data Ingestion for Project ATLAS-X...")

    # 1. Load CSVs
    print("--- Loading CSVs ---")
    train_trans = pd.read_csv(raw_path / "train_transaction.csv")
    train_id = pd.read_csv(raw_path / "train_identity.csv")

    # 2. Left Join on TransactionID
    print("--- Merging Dataframes ---")
    train = pd.merge(train_trans, train_id, on='TransactionID', how='left')
    
    # 3. Memory Optimization
    print("--- Reducing Memory Usage ---")
    train = reduce_mem_usage(train)

    # 4. Save to Parquet
    output_file = processed_path / "train_merged.parquet"
    print(f"--- Saving to {output_file} ---")
    train.to_parquet(output_file, engine='pyarrow')

    print("\n✅ Ingestion Complete!")
    print(f"Final Shape: {train.shape}")
    print(f"Total Columns: {len(train.columns)}")

if __name__ == "__main__":
    start_time = time.time()
    ingest_data()
    print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")