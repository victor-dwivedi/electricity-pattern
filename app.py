import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Function to perform clustering
def perform_clustering(df):
    # Feature Engineering: Extract hour from timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    
    # Group by hour of the day to find average energy usage at each hour across the month
    hourly_usage = df.groupby('hour')['energy_usage'].mean().reset_index()

    # Standardizing the data (only energy usage is used for clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(hourly_usage[['energy_usage']])

    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # GMM Clustering
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)

    # Adding cluster labels to the hourly_usage dataframe
    hourly_usage['kmeans_cluster'] = kmeans_labels
    hourly_usage['gmm_cluster'] = gmm_labels
    
    return hourly_usage, kmeans_labels, gmm_labels, X_scaled

# Streamlit app
st.title("Energy Usage Clustering")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Check if the DataFrame contains the necessary columns
    if 'timestamp' in df.columns and 'energy_usage' in df.columns:
        hourly_usage, kmeans_labels, gmm_labels, X_scaled = perform_clustering(df)
        
        # Visualizing the K-Means clusters
        plt.figure(figsize=(10, 6))
        plt.scatter(hourly_usage['hour'], hourly_usage['energy_usage'], c=hourly_usage['kmeans_cluster'], cmap='viridis')
        plt.title('K-Means Clustering of Energy Consumption Patterns by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Energy Usage')
        plt.xticks(range(24))
        plt.grid(True)
        st.pyplot(plt)

        # Visualizing the GMM clusters
        plt.figure(figsize=(10, 6))
        plt.scatter(hourly_usage['hour'], hourly_usage['energy_usage'], c=hourly_usage['gmm_cluster'], cmap='plasma')
        plt.title('GMM Clustering of Energy Consumption Patterns by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Energy Usage')
        plt.xticks(range(24))
        plt.grid(True)
        st.pyplot(plt)

        # Calculate silhouette scores
        kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
        gmm_silhouette = silhouette_score(X_scaled, gmm_labels)

        # Display silhouette scores
        st.write(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")
        st.write(f"GMM Silhouette Score: {gmm_silhouette:.4f}")
    else:
        st.error("The CSV file must contain 'timestamp' and 'energy_usage' columns.")
