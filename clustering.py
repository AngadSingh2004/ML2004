# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load the Iris dataset
iris = datasets.load_iris()
data = iris.data
target = iris.target
df = pd.DataFrame(data, columns=iris.feature_names)
df['target'] = target

# Display first few rows of the dataset
df.head()

# Function to evaluate clustering performance
def evaluate_clustering(labels, data):
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    return silhouette, davies_bouldin

# Pre-processing techniques
def preprocess_data(data, method="standard"):
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        return data
    
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Apply PCA for visualization purposes (optional)
def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data

# Clustering algorithms to be tested
def apply_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    inertia = kmeans.inertia_
    return labels, inertia

def apply_hierarchical(data, n_clusters, linkage="ward"):
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = hc.fit_predict(data)
    return labels

def apply_meanshift(data):
    ms = MeanShift()
    labels = ms.fit_predict(data)
    return labels

# Define parameters for the study
preprocessing_methods = ["standard", "minmax"]
n_clusters_list = [2, 3, 4, 5]
linkage_methods = ["ward", "complete", "average"]

# Data storage for evaluation results
results = []

# Perform clustering with different methods and evaluate
for method in preprocessing_methods:
    processed_data = preprocess_data(data, method)
    
    for n_clusters in n_clusters_list:
        # K-Means Clustering
        kmeans_labels, inertia = apply_kmeans(processed_data, n_clusters)
        silhouette, davies_bouldin = evaluate_clustering(kmeans_labels, processed_data)
        results.append(["KMeans", method, n_clusters, silhouette, davies_bouldin, inertia])
        
        # Hierarchical Clustering (different linkage methods)
        for linkage in linkage_methods:
            hierarchical_labels = apply_hierarchical(processed_data, n_clusters, linkage)
            silhouette, davies_bouldin = evaluate_clustering(hierarchical_labels, processed_data)
            results.append(["Hierarchical-" + linkage, method, n_clusters, silhouette, davies_bouldin, None])

    # Mean Shift Clustering
    meanshift_labels = apply_meanshift(processed_data)
    silhouette, davies_bouldin = evaluate_clustering(meanshift_labels, processed_data)
    results.append(["MeanShift", method, "auto", silhouette, davies_bouldin, None])

# Convert results to DataFrame for better readability
results_df = pd.DataFrame(results, columns=["Algorithm", "Preprocessing", "Clusters", "Silhouette Score", "Davies-Bouldin Score", "Inertia"])
print(results_df)

# Visualization of results
# Plotting Silhouette Score Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="Algorithm", y="Silhouette Score", hue="Preprocessing")
plt.title("Silhouette Score Comparison")
plt.xticks(rotation=45)
plt.show()

# Plotting Davies-Bouldin Score Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="Algorithm", y="Davies-Bouldin Score", hue="Preprocessing")
plt.title("Davies-Bouldin Score Comparison")
plt.xticks(rotation=45)
plt.show()

# Optional: PCA for cluster visualization (2D plot)
pca_data = apply_pca(processed_data, n_components=2)
kmeans_labels, _ = apply_kmeans(processed_data, n_clusters=3)  # Example with KMeans and 3 clusters

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=kmeans_labels, palette="viridis", s=60)
plt.title("PCA Projection of Clusters (KMeans, 3 Clusters)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()