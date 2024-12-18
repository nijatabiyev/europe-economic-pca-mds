import pandas as pd
from sklearn.decomposition import PCA

def preprocess_data(data):
    """
    Preprocess the dataset for PCA and MDS.
    Steps:
    - Remove non-numeric columns.
    - Handle missing values by dropping rows with NaNs.
    - Normalize the data (important for PCA and MDS).
    """
    # Keep only numeric columns
    numeric_data = data.select_dtypes(include=["float64", "int64"])
    
    # Drop rows with missing values
    clean_data = numeric_data.dropna()
    
    # Normalize the data (mean = 0, std = 1)
    normalized_data = (clean_data - clean_data.mean()) / clean_data.std()
    
    return normalized_data

def run_pca(data, n_components=2):
    """
    Perform PCA on the dataset.
    Args:
        data: Preprocessed dataset (numeric, normalized).
        n_components: Number of principal components to retain.
    Returns:
        Transformed data in the reduced dimensions.
    """
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    
    # Variance explained by each component
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained Variance Ratio (PCA): {explained_variance}")
    
    return pca_data, explained_variance
