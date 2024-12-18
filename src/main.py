import pandas as pd
import os
from analysis import preprocess_data, run_pca

# Step 1: Define the data path
# Use an absolute path to avoid ambiguity
data_path = r"C:\Users\nijat\europe-economic-pca-mds\data\europe_data.csv"

# Verify the file exists
if not os.path.exists(data_path):
    print(f"Error: File not found at {data_path}")
    exit()

# Step 2: Load the data
data = pd.read_csv(data_path)
print("Data Preview:")
print(data.head())

# Step 3: Preprocess the data
preprocessed_data = preprocess_data(data)
print("\nPreprocessed Data:")
print(preprocessed_data.head())

# Step 4: Perform PCA
pca_data, explained_variance = run_pca(preprocessed_data, n_components=2)
print("\nExplained Variance Ratio (PCA):")
print(explained_variance)

print("\nPCA Transformed Data (First 5 Rows):")
print(pca_data[:5])

import matplotlib.pyplot as plt

# Visualize PCA results
def plot_pca(pca_data):
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c='blue', alpha=0.7, edgecolors='k')
    plt.title("PCA Result: 2D Projection", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.grid(True)
    plt.show()

# Call the plotting function
plot_pca(pca_data)

# Perform PCA with more components
pca_data, explained_variance = run_pca(preprocessed_data, n_components=3)

print("\nExplained Variance Ratio (PCA):")
print(explained_variance)

print("\nCumulative Explained Variance:")
print(explained_variance.cumsum())  # To see cumulative variance explained
