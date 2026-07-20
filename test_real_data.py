import numpy as np
import pandas as pd
import feature_engineering as fe
import time

def test_on_real_data():
    print("Loading real Altamira Centered Matrix (this might take a few seconds)...")
    
    # We will test on the precomputed centered matrix from Phase 1
    file_path = "Preprocess/Altamira_NDVI_CenteredMatrix.parquet"
    
    t0 = time.time()
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}. Make sure preprocessing ran successfully.")
        return
        
    t1 = time.time()
    print(f"Loaded in {t1 - t0:.2f} seconds.")
    print(f"Original shape (Pixels, Dates): {df.shape}")
    
    # Convert dummy values (-99999) back to NaN before calculating
    df = df.replace(-99999, np.nan)
    centered_vals = df.values
    
    print("\nStarting Feature Engineering Pipeline...")
    t2 = time.time()
    feature_tensor = fe.build_time_aware_features(centered_vals, window_size=3)
    t3 = time.time()
    
    print(f"Pipeline finished in {t3 - t2:.2f} seconds.")
    print(f"Output Tensor Shape (Pixels, Dates, Features): {feature_tensor.shape}")
    
    # Calculate approximate memory size
    mem_size_mb = feature_tensor.nbytes / (1024 * 1024)
    print(f"Tensor Memory Footprint: ~{mem_size_mb:.2f} MB")
    
    print("\n--- Sample Features for First Valid Pixel ---")
    # Find a pixel with actual data (not all NaNs/0s)
    for i in range(feature_tensor.shape[0]):
        if np.any(feature_tensor[i, :, 0] != 0):
            pixel_data = feature_tensor[i]
            print(f"Pixel index {i} data for first 5 dates:")
            for d in range(5):
                print(f"  Date {d}: [X={pixel_data[d,0]:.4f}, V={pixel_data[d,1]:.4f}, A={pixel_data[d,2]:.4f}, Mean={pixel_data[d,3]:.4f}, Var={pixel_data[d,4]:.4f}]")
            break

if __name__ == '__main__':
    test_on_real_data()
