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
