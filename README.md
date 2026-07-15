# DynaLand
File base repository of the article "Integrating Unsupervised Machine Intelligence and Anomaly Detection for Spatial-Temporal Dynamics Mapping using Remote Sensing Image Series"

Anomaly Detection code flow:

1. Select coordinates and build our ImageCollection
- Sentinel Multispectral Images have 10 meters of spatial resolution and 16 bands.
- Band selection and Vegetation Indexes apply (NDVI and NDWI)
- Plot the time variation of the Vegetation Indexes

2. Reduce function building and passing band values to Pandas DataFrame
- Median subtraction to get the best central trend
- Function to extract pixel values for a Numpy Array
- Dates (columns) x Lat/Lon (multi index)

3. Machine Learning Anomaly Detection methods application
- Anomalies = -1; Regular = 1;
- At this point, we have to optimize our training dataset. For that, we put a proportional bound ($\alpha$) to the standard deviation of mean, to extract the most likely probable values.
- Vectorizing the DataFrames values to calculate the mean and standard deviation
- Input $\alpha$ value to optimize the dataset, returning one regular array:
    - upper bound = $mean + \alpha*std$
    - lower bound = $mean - \alpha*std$
    - array_reg = [lower bound:upper bound]
- For One-Class SVM, input a $\beta$ value to reduce the array size and optimize processing time of training without lettering.
- Training Anomaly Detection methods
- Define our statistic functions:
    - transitions -> 1 to -1 or -1 to 1 (== changes);
    - mantem_normal -> 1 to 1;
    - mantem_anomalia -> -1 to -1;
    - contador_reg -> count all regular pixels;
    - contador_anomaly -> count all anomaly pixels;
    - p_valor -> calculate the p-value of changes;
- Define the function <strong>OCSVM</strong> to map in DataDrame
- For each parameter of anomaly methods, run the for loop to create its DataFrame
- Build a DataFrame of anomalies in period of interest
- Save Tiff from DataFrame

## File Explanations

Here is a summary of the key files in this repository:

### Core Analysis Notebooks & Scripts
These contain the primary exploration and model training logic for the three study regions described in the article:
- **Altamira (MODIS NDVI)**: [Altamira_Modis_script.ipynb](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/Altamira_Modis_script.ipynb) / [Altamira_Modis_script.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/Altamira_Modis_script.py)
- **Brumadinho (Sentinel-2 NDWI)**: [Brumadinho_Sentinel_script.ipynb](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/Brumadinho_Sentinel_script.ipynb) / [Brumadinho_Sentinel_script.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/Brumadinho_Sentinel_script.py)
- **Mariana (Landsat-8 GVMI)**: [Mariana_Landsat_script.ipynb](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/Mariana_Landsat_script.ipynb) / [Mariana_Landsat_script.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/Mariana_Landsat_script.py)
- **Synthetic Validation**: [V9_Syntethic_Image_Validation.ipynb](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/V9_Syntethic_Image_Validation.ipynb) / [V9_Syntethic_Image_Validation.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/V9_Syntethic_Image_Validation.py) — Validates the anomaly detection methodology using synthetic time-series images.

### Reproducibility Pipelines
These scripts allow you to run the pipelines systematically and log results using MLflow:
- [run_preprocessing.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/run_preprocessing.py): Runs baseline data ingestion, cloud filtering, median-centering, and envelope selection for all three regions.
- [Altamira_Modis_repro.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/Altamira_Modis_repro.py), [Brumadinho_Sentinel_repro.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/Brumadinho_Sentinel_repro.py), [Mariana_Landsat_repro.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/Mariana_Landsat_repro.py): Scripts to run parameter sweeps (e.g., number of estimators for Isolation Forest, $\nu$ for One-Class SVM) under both `leak_free=True` and `leak_free=False` settings, logging results to MLflow.

### Histogram Analysis
Notebooks and generated scripts for visualizing frequency distributions of remote sensing indices and calculating statistical thresholds:
- **Altamira**: [histograma_altamira-V2.ipynb](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/histograma_altamira-V2.ipynb) / [histograma_altamira-V2.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/histograma_altamira-V2.py)
- **Brumadinho**: [histograma_brumadinho_V2.ipynb](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/histograma_brumadinho_V2.ipynb) / [histograma_brumadinho_V2.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/histograma_brumadinho_V2.py)
- **Mariana**: [histograma_Mariana_V2.ipynb](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/histograma_Mariana_V2.ipynb) / [histograma_Mariana_V2.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/histograma_Mariana_V2.py)

### Shared Modules & Configuration
- [repro_utils.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/repro_utils.py): Core helper functions for loading datasets, preprocessing time series, calculating metrics, wrapping models, logging to MLflow, and exporting results.
- [requirements.txt](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/requirements.txt): Environment dependencies.

### Datasets, Reference PDF, and Outputs
- `data_Altamira_ndvi.parquet` & `data_Altamira_SummaryQA.parquet`: Cached Parquet datasets representing NDVI and QA information for the Altamira MODIS analysis.
- [sustainability-15-04725.pdf](file:///c:/Users/dani9/.gemini/antigravity/scratch/TimeSeriesProject/sustainability-15-04725.pdf): The original research paper describing the methodology and results.
- `Tiff/`: Directory containing output GeoTIFF files generated from the anomaly detection models for visualization in GIS software.
