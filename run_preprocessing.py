import repro_utils

def main():
    print("Starting baseline data ingestion, cloud filtering, centering, and envelope selection...")
    alpha = 0.5
    
    # 1. Altamira (MODIS NDVI)
    repro_utils.run_and_log_preprocessing('Altamira', 'ndvi', alpha=alpha)
    
    # 2. Brumadinho (Sentinel-2 NDWI)
    repro_utils.run_and_log_preprocessing('Brumadinho', 'ndwi', alpha=alpha)
    
    # 3. Mariana (Landsat-8 GVMI)
    repro_utils.run_and_log_preprocessing('Mariana', 'gvmi', alpha=alpha)
    
    print("\nAll baseline preprocessing pipelines executed and logged to MLflow successfully!")

if __name__ == '__main__':
    main()
