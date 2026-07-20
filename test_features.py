import numpy as np
import feature_engineering as fe

def test_pipeline():
    # Create a dummy test sequence: 2 pixels, 5 timesteps
    # Pixel 0: [1, 2, 4, 7, 11] -> Velocity should be [NaN, 1, 2, 3, 4] -> Acceleration [NaN, NaN, 1, 1, 1]
    # Pixel 1: [10, 8, 6, 4, 2] -> Velocity should be [NaN, -2, -2, -2, -2] -> Acceleration [NaN, NaN, 0, 0, 0]
    dummy_data = np.array([
        [1.0, 2.0, 4.0, 7.0, 11.0],
        [10.0, 8.0, 6.0, 4.0, 2.0]
    ])
    
    print("--- DUMMY DATA ---")
    print(dummy_data)
    
    tensor = fe.build_time_aware_features(dummy_data, window_size=3)
    
    print("\n--- OUTPUT TENSOR SHAPE ---")
    print(tensor.shape)  # Should be (2, 5, 5)
    
    print("\n--- PIXEL 0 FEATURES OVER TIME ---")
    for t in range(5):
        print(f"Time {t}: [X={tensor[0, t, 0]}, V={tensor[0, t, 1]}, A={tensor[0, t, 2]}, Mean={tensor[0, t, 3]:.2f}, Var={tensor[0, t, 4]:.2f}]")
        
    print("\n--- PIXEL 1 FEATURES OVER TIME ---")
    for t in range(5):
        print(f"Time {t}: [X={tensor[1, t, 0]}, V={tensor[1, t, 1]}, A={tensor[1, t, 2]}, Mean={tensor[1, t, 3]:.2f}, Var={tensor[1, t, 4]:.2f}]")

if __name__ == '__main__':
    test_pipeline()
