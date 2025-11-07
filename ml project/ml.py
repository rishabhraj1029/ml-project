# project_main.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import trim_mean
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import umap
import wandb
import warnings
warnings.filterwarnings('ignore')

# Optional time-series clustering
try:
    from tslearn.clustering import TimeSeriesKMeans
    TSLEARN_AVAILABLE = True
except Exception as e:
    TSLEARN_AVAILABLE = False
    print("tslearn not available; time-series clustering will be skipped unless installed.")

# -------------------------
# USER: update these dataset paths to where you've downloaded files
DATASETS = {
    "METR-LA": "data/metr_la_speed.csv",        # expected: timestamp, sensor_id, speed, lat, lon
    "PEMS-BAY": "data/pems_bay.csv",            # similar format
    "NYC-TAXI": "data/nyc_taxi_agg.csv",
    "UBER-MOVEMENT": "data/uber_movement.csv"
}
# -------------------------

# W&B setup: fill with your entity/project
WANDB_ENTITY = "<your-wandb-entity>"  # or None
WANDB_PROJECT = "traffic-clustering"

def load_generic_timeseries(path, time_col='timestamp'):
    df = pd.read_csv(path, parse_dates=[time_col])
    return df

# Example generic loader - you may need to adapt per dataset
def aggregate_to_timeseries(df, id_col='sensor_id', time_col='timestamp', value_col='speed', freq='30min'):
    """
    Returns a pivoted timeseries DataFrame: rows=time, cols=ids (sensor/time series)
    """
    df = df[[time_col, id_col, value_col]].copy()
    df = df.dropna(subset=[time_col, id_col])
    df = df.set_index(time_col)
    # resample per sensor
    agg = df.groupby(id_col).resample(freq)[value_col].mean().reset_index()
    pivot = agg.pivot(index=time_col, columns=id_col, values=value_col)
    return pivot

# -------------------------
# Descriptive statistics and cleaning functions
def descriptive_statistics(ts_df):
    # ts_df: DataFrame with columns = sensors, index = timestamps
    stats_df = pd.DataFrame(index=ts_df.columns)
    stats_df['count'] = ts_df.count()
    stats_df['mean'] = ts_df.mean()
    stats_df['median'] = ts_df.median()
    stats_df['std'] = ts_df.std()
    stats_df['min'] = ts_df.min()
    stats_df['max'] = ts_df.max()
    stats_df['skew'] = ts_df.skew()
    stats_df['kurtosis'] = ts_df.kurtosis()
    stats_df['25p'] = ts_df.quantile(0.25)
    stats_df['75p'] = ts_df.quantile(0.75)
    stats_df['IQR'] = stats_df['75p'] - stats_df['25p']
    return stats_df

def detect_outliers_box(ts_series):
    q1 = ts_series.quantile(0.25)
    q3 = ts_series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (ts_series < lower) | (ts_series > upper)

def trim_series(series, trim_fraction=0.05):
    n = len(series)
    k = int(np.floor(trim_fraction * n))
    if k == 0:
        return series.mean(), series.median(), series.std()
    sorted_vals = np.sort(series.dropna())
    trimmed = sorted_vals[k: n - k]
    return trimmed.mean(), np.median(trimmed), np.std(trimmed)

# -------------------------
# Feature engineering for clustering (per-sensor features)
def engineer_features(ts_df):
    # Input: time x sensors
    features = {}
    for col in ts_df.columns:
        s = ts_df[col]
        # basic stats
        features[col] = {
            'mean': s.mean(),
            'median': s.median(),
            'std': s.std(),
            'min': s.min(),
            'max': s.max(),
            '25p': s.quantile(0.25),
            '75p': s.quantile(0.75),
            'peak_to_offpeak': s.mean() - s.median(),
            'skew': s.skew()
        }
        # add hourly peak magnitude
        try:
            hourly = s.groupby(s.index.hour).mean()
            features[col]['hour_of_max'] = hourly.idxmax()
            features[col]['hour_max_val'] = hourly.max()
        except:
            features[col]['hour_of_max'] = np.nan
            features[col]['hour_max_val'] = np.nan
    feat_df = pd.DataFrame.from_dict(features, orient='index')
    return feat_df

# -------------------------
# Clustering & evaluation utilities
def evaluate_clustering(X, labels):
    res = {}
    # handle trivial cluster cases
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        res['silhouette'] = np.nan
        res['calinski_harabasz'] = np.nan
        res['davies_bouldin'] = np.nan
        res['mean_intra_dist'] = np.nan
        res['mean_inter_dist'] = np.nan
        res['cluster_size_variance'] = np.var(np.bincount(labels + (labels.min()*-1)))
        return res
    try:
        res['silhouette'] = silhouette_score(X, labels)
    except:
        res['silhouette'] = np.nan
    try:
        res['calinski_harabasz'] = calinski_harabasz_score(X, labels)
    except:
        res['calinski_harabasz'] = np.nan
    try:
        res['davies_bouldin'] = davies_bouldin_score(X, labels)
    except:
        res['davies_bouldin'] = np.nan

    dists = pairwise_distances(X)
    # intra cluster
    intra = []
    inter = []
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        if len(idx) <= 1: continue
        intra.append(dists[np.ix_(idx, idx)].mean())
    # inter cluster: distances between centroids
    centroids = []
    for k in np.unique(labels):
        centroids.append(X[labels == k].mean(axis=0))
    if len(centroids) > 1:
        centroids = np.array(centroids)
        inter = pairwise_distances(centroids)
        res['mean_inter_dist'] = inter[np.triu_indices_from(inter, k=1)].mean()
    else:
        res['mean_inter_dist'] = np.nan
    res['mean_intra_dist'] = np.nanmean(intra) if len(intra)>0 else np.nan
    # cluster size variance
    counts = np.array([np.sum(labels == k) for k in np.unique(labels)])
    res['cluster_size_variance'] = counts.var()
    return res

def run_and_evaluate(X, method_name, **kwargs):
    # X: numpy array (n_samples x n_features)
    if method_name == 'kmeans':
        n_clusters = kwargs.get('n_clusters', 4)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)
    elif method_name == 'agglo':
        n_clusters = kwargs.get('n_clusters', 4)
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=kwargs.get('linkage','ward'))
        labels = model.fit_predict(X)
    elif method_name == 'dbscan':
        model = DBSCAN(eps=kwargs.get('eps', 0.5), min_samples=kwargs.get('min_samples', 5))
        labels = model.fit_predict(X)
    elif method_name == 'spectral':
        n_clusters = kwargs.get('n_clusters', 4)
        model = SpectralClustering(n_clusters=n_clusters, random_state=42, assign_labels='kmeans')
        labels = model.fit_predict(X)
    elif method_name == 'ts_kmeans' and TSLEARN_AVAILABLE:
        n_clusters = kwargs.get('n_clusters', 4)
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
        labels = model.fit_predict(kwargs['series'])  # expects series array
    else:
        raise ValueError("Unknown method or missing data for timeseries method")
    metrics = evaluate_clustering(X if method_name!='ts_kmeans' else kwargs['series'], labels)
    return model, labels, metrics

# -------------------------
# Main pipeline per dataset
def process_dataset(path, dataset_name):
    print(f"Processing dataset: {dataset_name}")
    raw = load_generic_timeseries(path)
    # NOTE: you may need to adapt column names here per dataset
    # Assume standard columns: timestamp, sensor_id, speed
    # If dataset already pivoted, skip aggregation
    if {'timestamp','sensor_id','speed'}.issubset(raw.columns):
        ts = aggregate_to_timeseries(raw, id_col='sensor_id', value_col='speed', time_col='timestamp', freq='30min')
    else:
        # if rows are per time and columns per sensor already:
        ts = raw.copy()
        ts.index = pd.to_datetime(ts.index)

    print("Time series shape:", ts.shape)
    # Data quality
    missing_frac = ts.isna().mean().mean()
    print(f"Missing fraction (overall): {missing_frac:.4f}")

    # Descriptive stats
    stats_df = descriptive_statistics(ts)
    print("Descriptive stats (top 5 sensors):")
    print(stats_df.head())

    # Outlier detection example for one sensor
    example_col = ts.columns[0]
    out_mask = detect_outliers_box(ts[example_col])
    print(f"Sensor {example_col} outlier count:", out_mask.sum())

    # Missing value handling: per-sensor interpolation then fill by column mean
    ts_imputed = ts.copy()
    ts_imputed = ts_imputed.interpolate(method='time', limit=6, axis=0)
    ts_imputed = ts_imputed.fillna(ts_imputed.mean())

    # Distribution pattern: plot single sensor histogram (you can extend)
    # Compute requested percentiles and trimmed stats for one sample sensor
    s = ts_imputed[example_col].dropna()
    sample_mean = s.mean()
    p25 = s.quantile(0.25)
    p75 = s.quantile(0.75)
    median = s.median()
    trimmed_mean, trimmed_median, trimmed_std = trim_series(s.values, trim_fraction=0.05)

    print("Sample stats:", sample_mean, p25, p75, median, trimmed_mean, trimmed_std)

    # Feature engineering
    feat_df = engineer_features(ts_imputed)
    print("Feature table shape:", feat_df.shape)

    # Standardize features
    X = feat_df.fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Dimensionality reduction for visualization (UMAP)
    reducer = umap.UMAP(random_state=42)
    emb = reducer.fit_transform(Xs)

    # Run clustering methods and log results to wandb
    experiments_results = {}
    methods = ['kmeans','agglo','dbscan']
    if TSLEARN_AVAILABLE:
        methods.append('ts_kmeans')

    # Initialize W&B run
    wandb_run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=f"{dataset_name}_experiment", reinit=True)
    for method in methods:
        if method == 'kmeans' or method == 'agglo' or method == 'spectral':
            for k in [3,4,5,6]:
                model, labels, metrics = run_and_evaluate(Xs, method, n_clusters=k)
                key = f"{method}_k{k}"
                experiments_results[key] = metrics
                # log
                wandb.log({f"{dataset_name}/{key}/{m}": v for m,v in metrics.items()})
                # plot embedding colored by labels
                plt.figure(figsize=(6,4))
                plt.scatter(emb[:,0], emb[:,1], c=labels, cmap='tab10', s=10)
                plt.title(f"{dataset_name} {key}")
                plt.tight_layout()
                wandb.log({f"{dataset_name}/{key}/umap": wandb.Image(plt)})
                plt.close()
        elif method == 'dbscan':
            for eps in [0.5, 1.0, 2.0]:
                model, labels, metrics = run_and_evaluate(Xs, method, eps=eps, min_samples=3)
                key = f"{method}_eps{eps}"
                experiments_results[key] = metrics
                wandb.log({f"{dataset_name}/{key}/{m}": v for m,v in metrics.items()})
        elif method == 'ts_kmeans' and TSLEARN_AVAILABLE:
            # Build series for tslearn (sensors x time)
            series = ts_imputed.T.fillna(0).values  # shape sensors x time
            series = series.reshape(series.shape[0], series.shape[1], 1)
            for k in [3,4,5]:
                model, labels, metrics = run_and_evaluate(None, method, n_clusters=k, series=series)
                key = f"{method}_k{k}"
                experiments_results[key] = metrics
                wandb.log({f"{dataset_name}/{key}/{m}": v for m,v in metrics.items()})
    wandb_run.finish()

    return {
        'raw_stats': stats_df,
        'features': feat_df,
        'experiments': experiments_results
    }

if __name__ == "__main__":
    # initialize W&B at script start (this run will serve as aggregator)
    wandb.login()
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"WARNING: dataset file not found: {path}. Please download and set path.")
            continue
        result = process_dataset(path, name)
        # Save artifacts
        out_dir = "output"
        os.makedirs(out_dir, exist_ok=True)
        result['raw_stats'].to_csv(os.path.join(out_dir, f"{name}_stats.csv"))
        result['features'].to_csv(os.path.join(out_dir, f"{name}_features.csv"))
        print(f"Saved stats/features for {name}")
