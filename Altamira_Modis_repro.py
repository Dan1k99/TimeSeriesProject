import repro_utils

def main():
    dataset_name = 'Altamira'
    band_name = 'ndvi'
    
    print(f"Starting experiments for {dataset_name} ({band_name.upper()})...")
    
    # Pre-download/cache raw data first
    df_raw = repro_utils.load_raw_data(dataset_name, band_name)
    print(f"Raw data loaded. Shape: {df_raw.shape}")
    
    # 1. Sweep Isolation Forest parameters
    alpha_if = 1.0
    beta_if = None
    model_type_if = 'IsolationForest'
    n_estimators_list = [20, 40, 60, 80, 100]
    
    for leak_free in [False, True]:
        print(f"\n--- Running Isolation Forest (leak_free={leak_free}) ---")
        for n_est in n_estimators_list:
            model_params = {'n_estimators': n_est}
            repro_utils.log_experiment_to_mlflow(
                dataset_name, band_name, alpha_if, beta_if, model_type_if, model_params, leak_free
            )
            
    # 2. Sweep One-Class SVM parameters
    alpha_oc = 0.5
    beta_oc = 0.005
    model_type_oc = 'OneClassSVM'
    nu_list = [0.025, 0.05, 0.1]
    
    for leak_free in [False, True]:
        print(f"\n--- Running One-Class SVM (leak_free={leak_free}) ---")
        for nu in nu_list:
            model_params = {'nu': nu, 'kernel': 'rbf', 'gamma': 'auto'}
            repro_utils.log_experiment_to_mlflow(
                dataset_name, band_name, alpha_oc, beta_oc, model_type_oc, model_params, leak_free
            )
            
    print("\nAll Altamira MODIS experiments logged successfully to MLflow!")

if __name__ == '__main__':
    main()
