import os
import time
import datetime
import random
import numpy as np
import pandas as pd
import ee
import mlflow
from osgeo import gdal
from osgeo import osr
from scipy.stats import norm

# Initialize GDAL PROJ database path to avoid projection errors on Windows
PROJ_PATH = r"C:\Users\dani9\.gemini\antigravity\scratch\TimeSeriesProject\.venv\Lib\site-packages\osgeo\data\proj"
if os.path.exists(PROJ_PATH):
    os.environ['PROJ_LIB'] = PROJ_PATH

def init_ee():
    if not ee.data.is_initialized():
        PROJECT_IDS = ['889258893131', 'timeseriesproject-503021']
        for proj in PROJECT_IDS:
            try:
                ee.Initialize(project=proj)
                return
            except Exception:
                continue
        try:
            ee.Initialize()
        except Exception:
            ee.Authenticate()
            ee.Initialize()

def get_dataset_config(dataset_name, band_name):
    # Configurations for the three paper experiments
    if dataset_name == 'Altamira':
        geometria = ee.Geometry.Polygon([[
            [-55.3809, -7.5173],
            [-54.714, -7.5173],
            [-54.714, -8.005],
            [-55.3809, -8.005],
            [-55.3809, -7.5173]
        ]])
        return {
            'collection_name': 'MODIS/061/MOD13Q1',
            'start_date': '2010-01-01',
            'end_date': '2021-12-31',
            'scale': 250,
            'geometry': geometria,
            'select_bands': ['NDVI', 'SummaryQA'],
            'filter_fn': lambda img: img,  # We filter df later using SummaryQA
            'band_calc': lambda img: img.expression('NDVI / 10000', {'NDVI': img.select('NDVI')}).rename('ndvi')
        }
    elif dataset_name == 'Brumadinho':
        geometria = ee.Geometry.Polygon([[
            [-44.13576882693976, -20.14040952973489],
            [-44.107444699742494, -20.14040952973489],
            [-44.107444699742494, -20.11204199216084],
            [-44.13576882693976, -20.11204199216084],
            [-44.13576882693976, -20.14040952973489]
        ]])
        if band_name == 'ndvi':
            band_calc = lambda img: img.expression('(nir - red) / (nir + red)', {'nir': img.select('B8'), 'red': img.select('B4')}).rename('ndvi')
        else:
            band_calc = lambda img: img.expression('(nir - green) / (nir + green)', {'nir': img.select('B8'), 'green': img.select('B3')}).rename('ndwi')
        return {
            'collection_name': 'COPERNICUS/S2_HARMONIZED',
            'start_date': '2013-01-01',
            'end_date': '2021-12-31',
            'scale': 10,
            'geometry': geometria,
            'select_bands': ['B8', 'B4', 'B3'],
            'filter_fn': lambda col: col.filterMetadata('CLOUD_COVERAGE_ASSESSMENT', 'less_than', 20),
            'band_calc': band_calc
        }
    elif dataset_name == 'Mariana':
        geometria = ee.Geometry.Polygon([[
            [-43.49513553785783, -20.24537533206166],
            [-43.42406772779923, -20.24537533206166],
            [-43.42406772779923, -20.191089981448055],
            [-43.49513553785783, -20.191089981448055],
            [-43.49513553785783, -20.24537533206166]
        ]])
        if band_name == 'gvmi':
            band_calc = lambda img: img.expression('(nir - swir2 + 0.12) / (nir + swir2 + 0.12)', {'nir': img.select('B5'), 'swir2': img.select('B7')}).rename('gvmi')
        else:
            band_calc = lambda img: img.expression('(nir - green) / (nir + green)', {'nir': img.select('B5'), 'green': img.select('B3')}).rename('ndwi')
        return {
            'collection_name': 'LANDSAT/LC08/C02/T1_TOA',
            'start_date': '2013-01-01',
            'end_date': '2021-12-31',
            'scale': 30,
            'geometry': geometria,
            'select_bands': ['B5', 'B7', 'B3'],
            'filter_fn': lambda col: col.filterMetadata('CLOUD_COVER', 'less_than', 20),
            'band_calc': band_calc
        }
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

def load_raw_data(dataset_name, band_name):
    cache_file = f"data_{dataset_name}_{band_name}.parquet"
    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}...")
        try:
            df = pd.read_parquet(cache_file)
        except Exception:
            print("Applying PyArrow datetime fix for older pandas version...")
            import pyarrow.parquet as pq
            table = pq.read_table(cache_file)
            table = table.replace_schema_metadata()
            df = table.to_pandas()
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                df.set_index(['Latitude', 'Longitude'], inplace=True)
        return df

    print(f"Cache file {cache_file} not found. Querying Earth Engine (this may take a few minutes)...")
    init_ee()
    config = get_dataset_config(dataset_name, band_name)
    
    # Query GEE collection
    col = ee.ImageCollection(config['collection_name']) \
        .filterBounds(config['geometry']) \
        .filterDate(config['start_date'], config['end_date'])
    
    col = config['filter_fn'](col)
    
    # Map index calculation
    col_band = col.map(config['band_calc'])
    
    # Get image count and list of timestamps
    size = col_band.size().getInfo()
    print(f"Found {size} images.")
    
    # Get system timestamps
    timestamps = col.aggregate_array('system:time_start').getInfo()
    dates = [pd.to_datetime(ts, unit='ms') for ts in timestamps]
    
    # For Altamira, we also need SummaryQA
    qa_list_images = None
    if dataset_name == 'Altamira':
        col_qa = col.select('SummaryQA')
        qa_list_images = col_qa.toList(size)
    
    img_list = col_band.toList(size)
    
    # Get lats and lons from first image
    first_img = ee.Image(-99999).rename(band_name).blend(ee.Image(img_list.get(0))).addBands(ee.Image.pixelLonLat())
    coords = first_img.reduceRegion(
        reducer=ee.Reducer.toList(),
        geometry=config['geometry'],
        scale=config['scale'],
        bestEffort=True
    ).getInfo()
    
    lats = np.array(coords['latitude']).astype(float)
    lons = np.array(coords['longitude']).astype(float)
    
    # Build raw dictionary
    data_dict = {}
    
    # For Altamira we will track SummaryQA as well
    if dataset_name == 'Altamira':
        qa_values = []
        print("Downloading SummaryQA bands...")
        for j in range(size):
            img_qa = ee.Image(-99999).rename('SummaryQA').blend(ee.Image(qa_list_images.get(j)))
            qa_info = img_qa.reduceRegion(
                reducer=ee.Reducer.toList(),
                geometry=config['geometry'],
                scale=config['scale'],
                bestEffort=True
            ).getInfo()
            qa_values.append(np.array(qa_info['SummaryQA']).astype(float))
    
    print(f"Downloading {band_name} bands...")
    for j in range(size):
        date_str = dates[j].strftime('%Y-%m-%d')
        print(f"  Downloading image {j+1}/{size} for date {date_str}...")
        img = ee.Image(-99999).rename(band_name).blend(ee.Image(img_list.get(j)))
        info = img.reduceRegion(
            reducer=ee.Reducer.toList(),
            geometry=config['geometry'],
            scale=config['scale'],
            bestEffort=True
        ).getInfo()
        data_dict[dates[j]] = np.array(info[band_name]).astype(float)
    
    df = pd.DataFrame(data_dict)
    df['Latitude'] = lats
    df['Longitude'] = lons
    
    if dataset_name == 'Altamira':
        # Add SummaryQA info
        qa_df = pd.DataFrame(dict(zip(dates, qa_values)))
        qa_df['Latitude'] = lats
        qa_df['Longitude'] = lons
        qa_df.set_index(['Latitude', 'Longitude'], inplace=True)
        qa_df.to_parquet(f"data_{dataset_name}_SummaryQA.parquet")
        
    df.set_index(['Latitude', 'Longitude'], inplace=True)
    df.to_parquet(cache_file)
    print(f"Successfully cached data to {cache_file}!")
    return df

def calculate_metrics(df_pred):
    # Vectorized calculation of transition metrics and p-values
    metrics_df = pd.DataFrame(index=df_pred.index)
    arr = df_pred.values
    
    # 1. Anomalies and Regular counts
    metrics_df['Anomalias'] = np.sum(arr == -1, axis=1)
    metrics_df['Regular'] = np.sum(arr == 1, axis=1)
    
    # 2. Shifted array for transition metrics
    arr_t = arr[:, 1:]
    arr_t_minus_1 = arr[:, :-1]
    
    # transitions: sum of t + t-1 == 0
    metrics_df['Mudanças'] = np.sum((arr_t + arr_t_minus_1) == 0, axis=1)
    
    # mantem_normal: sum of t + t-1 == 2
    metrics_df['Permanece Regular'] = np.sum((arr_t + arr_t_minus_1) == 2, axis=1)
    
    # mantem_anomalia: sum of t + t-1 == -2
    metrics_df['Permanece Anomalia'] = np.sum((arr_t + arr_t_minus_1) == -2, axis=1)
    
    # Calculate transitions mean and std dev across all pixels to find z-score and p-value
    mud = metrics_df['Mudanças'].values
    mean_mud = np.mean(mud)
    std_mud = np.std(mud)
    
    metrics_df['media'] = mean_mud
    metrics_df['std'] = std_mud
    
    # Handle zero std dev to prevent nan
    if std_mud > 0:
        metrics_df['z'] = (mud - mean_mud) / std_mud
    else:
        metrics_df['z'] = 0.0
        
    metrics_df['p-valor'] = 1.0 - norm.cdf(metrics_df['z'].values)
    return metrics_df

def save_tiff_fromdf(df, bands, dummy, path_out):
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    
    lat = [idx[0] for idx in df.index]
    lon = [idx[1] for idx in df.index]
    
    ulat = np.unique(lat)
    ulon = np.unique(lon)
    ncols = len(ulon)
    nrows = len(ulat)
    nbands = len(bands)
    
    # Check that we have enough elements to estimate spacing
    if len(ulat) > 11:
        ys = ulat[11] - ulat[10]
    else:
        ys = ulat[1] - ulat[0] if len(ulat) > 1 else 0.002
        
    if len(ulon) > 11:
        xs = ulon[11] - ulon[10]
    else:
        xs = ulon[1] - ulon[0] if len(ulon) > 1 else 0.002
    
    arr = np.zeros([nbands, nrows, ncols], np.float32)
    refLat = np.max(ulat)
    refLon = np.min(ulon)
    
    for j in range(len(df)):
        posLin = np.int64(np.round((refLat - lat[j]) / ys))
        posCol = np.int64(np.round((lon[j] - refLon) / xs))
        
        # Clip indices to prevent index out of bounds
        posLin = np.clip(posLin, 0, nrows - 1)
        posCol = np.clip(posCol, 0, ncols - 1)
        
        for b in range(nbands):
            arr[b, posLin, posCol] = df.loc[df.index[j], bands[b]]
            
    transform = (np.min(ulon), xs, 0, np.max(ulat), 0, -ys)
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    
    driver = gdal.GetDriverByName('GTiff')
    outDs = driver.Create(path_out, ncols, nrows, nbands, gdal.GDT_Float32)
    outDs.SetGeoTransform(transform)
    outDs.SetProjection(target.ExportToWkt())

    for b in range(nbands):
        bandArr = np.copy(arr[b, :, :])
        outBand = outDs.GetRasterBand(b + 1)
        outBand.WriteArray(bandArr)
        outBand.FlushCache()
        outBand.SetNoDataValue(dummy)

    outDs = None
    print(f"Saved TIFF to {path_out}")

# Global memory cache for centering to prevent recomputations during sweeps
CENTERED_CACHE = {}

def get_centered_data(df_raw, dataset_name, band_name, leak_free):
    cache_key = (dataset_name, band_name, leak_free)
    if cache_key in CENTERED_CACHE:
        return CENTERED_CACHE[cache_key]
        
    print(f"Centering data for {dataset_name} {band_name} (leak_free={leak_free})...")
    default_dummy = -99999
    df_raw_nan = df_raw.replace(default_dummy, np.nan)
    raw_vals = df_raw_nan.values
    N_pixels, N_dates = raw_vals.shape
    
    t0 = time.time()
    if leak_free:
        centered_vals = np.full_like(raw_vals, np.nan)
        for j in range(N_dates):
            historical_median = np.nanmedian(raw_vals[:, :j+1], axis=1)
            centered_vals[:, j] = raw_vals[:, j] - historical_median
    else:
        Mediana = np.nanmedian(raw_vals, axis=1)
        centered_vals = raw_vals - Mediana[:, np.newaxis]
    t1 = time.time()
    print(f"Centering completed in {t1-t0:.2f}s.")
    
    CENTERED_CACHE[cache_key] = centered_vals
    return centered_vals

def run_experiment_pipeline(df_raw, centered_vals, alpha, beta, model_type, model_params, leak_free=True):
    # Enforce determinism
    random.seed(42)
    np.random.seed(42)
    
    N_pixels, N_dates = df_raw.shape
    print(f"Training {model_type} (leak_free={leak_free}) on {N_pixels} pixels across {N_dates} dates...")
    
    # Create prediction DataFrame
    df_pred = pd.DataFrame(1, index=df_raw.index, columns=df_raw.columns)
    
    t_start = time.time()
    
    if leak_free:
        # Fit classifier for each date
        for j in range(N_dates):
            if j % 20 == 0:
                print(f"  -> Training step {j}/{N_dates}...")
            # Envelope filtering: pool historical centered values up to date j
            hist_vals = centered_vals[:, :j+1].flatten()
            hist_vals = hist_vals[~np.isnan(hist_vals)]
            
            if len(hist_vals) > 0:
                mean_j = np.mean(hist_vals)
                std_j = np.std(hist_vals)
            else:
                mean_j = 0.0
                std_j = 1.0
                
            inf_lim = mean_j - alpha * std_j
            sup_lim = mean_j + alpha * std_j
            
            # Regular data envelope filtering
            train_data = hist_vals[(hist_vals > inf_lim) & (hist_vals < sup_lim)]
            
            if len(train_data) == 0:
                train_data = hist_vals if len(hist_vals) > 0 else np.array([0.0])
                
            # Downsample training data if it exceeds 30,000 samples for speed
            max_samples = 30000
            if model_type == 'OneClassSVM' and beta is not None:
                size_opt = int(beta * len(train_data))
                size_opt = min(size_opt, max_samples)
                if size_opt > 0:
                    dataind = np.random.choice(len(train_data), size=size_opt, replace=False)
                    train_data = train_data[dataind]
            else:
                if len(train_data) > max_samples:
                    dataind = np.random.choice(len(train_data), size=max_samples, replace=False)
                    train_data = train_data[dataind]
                    
            # Fit classifier on regular data
            if model_type == 'IsolationForest':
                from sklearn.ensemble import IsolationForest
                clf = IsolationForest(
                    n_estimators=model_params.get('n_estimators', 40),
                    random_state=42,
                    n_jobs=-1
                ).fit(train_data.reshape(-1, 1))
            elif model_type == 'OneClassSVM':
                from sklearn.svm import OneClassSVM
                clf = OneClassSVM(
                    nu=model_params.get('nu', 0.05),
                    kernel=model_params.get('kernel', 'rbf'),
                    gamma=model_params.get('gamma', 'auto')
                ).fit(train_data.reshape(-1, 1))
            else:
                raise ValueError(f"Unknown model type {model_type}")
                
            # Predict anomalies for date j
            current_vals = centered_vals[:, j]
            valid_idx = ~np.isnan(current_vals)
            if np.sum(valid_idx) > 0:
                pred = clf.predict(current_vals[valid_idx].reshape(-1, 1))
                df_pred.iloc[valid_idx, j] = pred
                
    else:
        # Leaky flow
        all_vals = centered_vals.flatten()
        all_vals = all_vals[~np.isnan(all_vals)]
        
        mean_all = np.mean(all_vals)
        std_all = np.std(all_vals)
        
        inf_lim = mean_all - alpha * std_all
        sup_lim = mean_all + alpha * std_all
        
        train_data = all_vals[(all_vals > inf_lim) & (all_vals < sup_lim)]
        
        if model_type == 'OneClassSVM' and beta is not None:
            size_opt = int(beta * len(train_data))
            dataind = np.random.choice(len(train_data), size=size_opt, replace=False)
            train_data = train_data[dataind]
            
        print(f"  Fitting {model_type} on {len(train_data)} training samples...")
        if model_type == 'IsolationForest':
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(
                n_estimators=model_params.get('n_estimators', 40),
                random_state=42,
                n_jobs=-1
            ).fit(train_data.reshape(-1, 1))
        elif model_type == 'OneClassSVM':
            from sklearn.svm import OneClassSVM
            clf = OneClassSVM(
                nu=model_params.get('nu', 0.05),
                kernel=model_params.get('kernel', 'rbf'),
                gamma=model_params.get('gamma', 'auto')
            ).fit(train_data.reshape(-1, 1))
            
        print(f"  Model fitted. Running predictions on {N_dates} dates...")
        # Predict on each column
        for j in range(N_dates):
            if j % 20 == 0:
                print(f"  -> Predicting step {j}/{N_dates}...")
            current_vals = centered_vals[:, j]
            valid_idx = ~np.isnan(current_vals)
            if np.sum(valid_idx) > 0:
                pred = clf.predict(current_vals[valid_idx].reshape(-1, 1))
                df_pred.iloc[valid_idx, j] = pred
                
    t_end = time.time()
    exec_time = t_end - t_start
    
    # Calculate metrics
    df_metrics = calculate_metrics(df_pred)
    
    return df_pred, df_metrics, exec_time

def log_experiment_to_mlflow(dataset_name, band_name, alpha, beta, model_type, model_params, leak_free):
    # Set experiment
    experiment_name = f"DynaLand_{dataset_name}_{band_name.upper()}"
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    run_name = f"{model_type}_{'leakfree' if leak_free else 'leaky'}"
    if model_type == 'IsolationForest':
        run_name += f"_est_{model_params.get('n_estimators')}"
    else:
        run_name += f"_nu_{model_params.get('nu')}"
        
    with mlflow.start_run(run_name=run_name) as run:
        # Load raw data
        df_raw = load_raw_data(dataset_name, band_name)
        
        # For Altamira, we filter columns (dates) where the mean SummaryQA < 1
        if dataset_name == 'Altamira':
            qa_df = pd.read_parquet("data_Altamira_SummaryQA.parquet")
            good_dates = qa_df.columns[qa_df.mean(axis=0) < 1]
            df_raw = df_raw[good_dates]
            
        # Get precomputed centered values (ignoring -99999 by treating it as NaN)
        centered_vals = get_centered_data(df_raw, dataset_name, band_name, leak_free)
        
        # Run pipeline
        df_pred, df_metrics, exec_time = run_experiment_pipeline(
            df_raw, centered_vals, alpha, beta, model_type, model_params, leak_free
        )
        
        # Log parameters
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("band", band_name)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("beta", beta)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("leak_free", leak_free)
        for k, v in model_params.items():
            mlflow.log_param(k, v)
            
        # Log summary metrics
        total_anomalies = int(df_metrics['Anomalias'].sum())
        total_regular = int(df_metrics['Regular'].sum())
        total_transitions = int(df_metrics['Mudanças'].sum())
        mean_transitions = float(df_metrics['media'].iloc[0])
        std_transitions = float(df_metrics['std'].iloc[0])
        
        mlflow.log_metric("total_anomalies", total_anomalies)
        mlflow.log_metric("total_regular", total_regular)
        mlflow.log_metric("total_transitions", total_transitions)
        mlflow.log_metric("mean_transitions", mean_transitions)
        mlflow.log_metric("std_transitions", std_transitions)
        mlflow.log_metric("execution_time_seconds", exec_time)
        
        # Save output TIFF
        tiff_dir = f"Tiff/{'leak_free' if leak_free else 'leaky'}/{model_type}"
        tiff_filename = f"{dataset_name}_{band_name.upper()}_{run_name}.tif"
        tiff_path = os.path.join(tiff_dir, tiff_filename)
        
        save_tiff_fromdf(df_metrics, ['Anomalias', 'p-valor'], -99999, tiff_path)
        
        # Log artifact
        mlflow.log_artifact(tiff_path)
        
        print(f"Logged run {run_name} to MLflow. Anomalies: {total_anomalies}, Transitions: {total_transitions}, Execution time: {exec_time:.2f}s")
        return df_metrics

def run_and_log_preprocessing(dataset_name, band_name, alpha=0.5):
    # Set experiment
    experiment_name = "DynaLand_Baseline_Reproduction"
    mlflow.set_experiment(experiment_name)
    
    run_name = f"Preprocess_{dataset_name}_{band_name.upper()}"
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n==================== Preprocessing: {dataset_name} ({band_name.upper()}) ====================")
        # 1. Load/download raw data
        df_raw = load_raw_data(dataset_name, band_name)
        
        # 2. Cloud/quality filtering (selection of dates)
        total_raw_dates = df_raw.shape[1]
        if dataset_name == 'Altamira':
            qa_df = pd.read_parquet("data_Altamira_SummaryQA.parquet")
            good_dates = qa_df.columns[qa_df.mean(axis=0) < 1]
            df_raw = df_raw[good_dates]
        
        unmasked_dates = df_raw.shape[1]
        print(f"Cloud/quality filtering: kept {unmasked_dates}/{total_raw_dates} dates.")
        
        # Replace default dummy with NaN for statistical calculations
        default_dummy = -99999
        df_raw_nan = df_raw.replace(default_dummy, np.nan)
        raw_vals = df_raw_nan.values
        N_pixels, N_dates = raw_vals.shape
        
        # 3. Median Centering (Trend Image & Centering)
        trend_image = np.nanmedian(raw_vals, axis=1)
        centered_vals = raw_vals - trend_image[:, np.newaxis]
        
        # 4. Statistical Envelope Selection
        all_vals = centered_vals.flatten()
        all_vals = all_vals[~np.isnan(all_vals)]
        
        mean_all = np.mean(all_vals)
        sigma = np.std(all_vals)
        
        inf_lim = mean_all - alpha * sigma
        sup_lim = mean_all + alpha * sigma
        
        regular_data = all_vals[(all_vals > inf_lim) & (all_vals < sup_lim)]
        
        # Log parameters
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("band", band_name)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("total_pixels", N_pixels)
        mlflow.log_param("raw_dates", total_raw_dates)
        mlflow.log_param("unmasked_dates", unmasked_dates)
        
        # Log metrics
        mlflow.log_metric("global_std_dev", sigma)
        mlflow.log_metric("envelope_inf_lim", inf_lim)
        mlflow.log_metric("envelope_sup_lim", sup_lim)
        mlflow.log_metric("regular_training_size", len(regular_data))
        
        # Save and log Trend Image as TIFF
        trend_df = pd.DataFrame(index=df_raw.index)
        trend_df['Trend_Median'] = trend_image
        trend_tiff_path = f"Preprocess/{dataset_name}_{band_name.upper()}_TrendImage.tif"
        save_tiff_fromdf(trend_df, ['Trend_Median'], default_dummy, trend_tiff_path)
        mlflow.log_artifact(trend_tiff_path)
        
        # Save Centered Matrix to Parquet
        centered_df = pd.DataFrame(centered_vals, index=df_raw.index, columns=df_raw.columns)
        # Restore dummy value -99999 for saving
        centered_df = centered_df.fillna(default_dummy)
        centered_parquet_path = f"Preprocess/{dataset_name}_{band_name.upper()}_CenteredMatrix.parquet"
        os.makedirs(os.path.dirname(centered_parquet_path), exist_ok=True)
        centered_df.to_parquet(centered_parquet_path)
        mlflow.log_artifact(centered_parquet_path)
        
        # Save Trend Image statistics to text file
        stats_path = f"Preprocess/{dataset_name}_{band_name.upper()}_TrendImage_Stats.txt"
        with open(stats_path, 'w') as f_stats:
            f_stats.write(f"Dataset: {dataset_name}\n")
            f_stats.write(f"Band: {band_name}\n")
            f_stats.write(f"Trend Image Mean: {np.nanmean(trend_image):.6f}\n")
            f_stats.write(f"Trend Image Std Dev: {np.nanstd(trend_image):.6f}\n")
            f_stats.write(f"Trend Image Min: {np.nanmin(trend_image):.6f}\n")
            f_stats.write(f"Trend Image Max: {np.nanmax(trend_image):.6f}\n")
            f_stats.write(f"Trend Image Median: {np.nanmedian(trend_image):.6f}\n")
        mlflow.log_artifact(stats_path)
        
        print(f"Logged Preprocessing for {dataset_name} to MLflow. Global Std: {sigma:.4f}, Regular count: {len(regular_data)}")
        return sigma
