import numpy as np
import pandas as pd
import warnings

def calculate_velocity(matrix):
    """
    Computes the velocity (first derivative) over time.
    V_t = X_t - X_{t-1}
    Returns a matrix of the same shape, with NaNs in the first column.
    """
    velocity = np.full_like(matrix, np.nan, dtype=float)
    velocity[:, 1:] = matrix[:, 1:] - matrix[:, :-1]
    return velocity

def calculate_acceleration(velocity_matrix):
    """
    Computes the acceleration (second derivative) over time.
    A_t = V_t - V_{t-1}
    Returns a matrix of the same shape, with NaNs in the first two columns.
    """
    acceleration = np.full_like(velocity_matrix, np.nan, dtype=float)
    acceleration[:, 1:] = velocity_matrix[:, 1:] - velocity_matrix[:, :-1]
    return acceleration

def calculate_rolling_stats(matrix, window_size=3):
    """
    Computes localized rolling mean and rolling variance over a sliding window.
    """
    # Use pandas for easy rolling window calculations along the time axis (axis=1)
    df = pd.DataFrame(matrix)
    
    # Calculate rolling mean and variance. min_periods=1 ensures we get some values 
    # even at the start of the sequence before the full window size is reached.
    rolling_mean = df.rolling(window=window_size, axis=1, min_periods=1).mean().values
    rolling_var = df.rolling(window=window_size, axis=1, min_periods=1).var().values
    
    # Variance might be NaN for the very first step (variance of 1 item), fill with 0
    rolling_var = np.nan_to_num(rolling_var, nan=0.0)
    
    return rolling_mean, rolling_var

def build_time_aware_features(centered_vals, window_size=3):
    """
    Combines the original centered data, velocity, acceleration, and rolling stats
    into a single 3D tensor suitable for Deep Learning models.
    
    Parameters:
    - centered_vals: 2D numpy array of shape (Num_Pixels, Num_Dates)
    
    Returns:
    - feature_tensor: 3D numpy array of shape (Num_Pixels, Num_Dates, 5)
                      Features are: [X_t, V_t, A_t, RollMean_t, RollVar_t]
    """
    print("Calculating velocity...")
    velocity = calculate_velocity(centered_vals)
    
    print("Calculating acceleration...")
    acceleration = calculate_acceleration(velocity)
    
    print(f"Calculating rolling stats (window={window_size})...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rolling_mean, rolling_var = calculate_rolling_stats(centered_vals, window_size)
        
    print("Stacking features into tensor...")
    # Stack along a new third dimension (Features)
    feature_tensor = np.stack([
        centered_vals, 
        velocity, 
        acceleration, 
        rolling_mean, 
        rolling_var
    ], axis=-1)
    
    # Handle NaNs from edge cases (e.g., first column has no velocity)
    # We replace NaNs with 0 to prevent Deep Learning models from crashing
    feature_tensor = np.nan_to_num(feature_tensor, nan=0.0)
    
    return feature_tensor
