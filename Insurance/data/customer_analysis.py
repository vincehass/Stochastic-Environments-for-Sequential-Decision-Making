import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the time series data
df = pd.read_csv("./Insurance/data/insurance_pricing_timeseries_dataset.csv")

# Summarize customer attributes
def summarize_customer_attributes(df):
    # Descriptive statistics for numeric attributes
    summary = df.describe()
    print("Descriptive statistics for customer attributes:")
    print(summary)

    # Plot distributions for customer features
    features = ['age', 'income', 'claim_history', 'competitor_price']
    plt.figure(figsize=(14, 10))
    for i, feature in enumerate(features):
        plt.subplot(2, 2, i + 1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()

    # Average CLTV score over time for each customer
    average_scores = df.groupby('customer_id')['score'].mean()
    print("\nAverage CLTV Score for each customer:")
    print(average_scores)

    # Plot the average CLTV score distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(average_scores, kde=True)
    plt.title('Distribution of Average CLTV Scores')
    plt.xlabel('Average CLTV Score')
    plt.ylabel('Frequency')
    plt.show()

# Clustering based on CLTV score
def cluster_customers(df, num_clusters=3):
    # Aggregate features for clustering (use average over time steps)
    customer_features = df.groupby('customer_id').agg({
        'age': 'mean',
        'income': 'mean',
        'claim_history': 'mean',
        'competitor_price': 'mean',
        'score': 'mean'
    }).reset_index()

    # Prepare data for clustering
    features_to_cluster = ['age', 'income', 'claim_history', 'competitor_price', 'score']
    X = customer_features[features_to_cluster]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    customer_features['cluster'] = kmeans.fit_predict(X_scaled)

    # Print cluster centers
    print("\nCluster centers (in standardized feature space):")
    print(kmeans.cluster_centers_)

    # Plot the clustering results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=customer_features['income'], y=customer_features['score'],
        hue=customer_features['cluster'], palette='viridis', s=100, alpha=0.7
    )
    plt.title('Customer Clustering Based on Income and CLTV Score')
    plt.xlabel('Income')
    plt.ylabel('Average CLTV Score')
    plt.legend(title='Cluster')
    plt.show()

    return customer_features

if __name__ == "__main__":
    # Load and analyze the data
    df = pd.read_csv("./Insurance/data/insurance_pricing_timeseries_dataset.csv")
    
    # Step 1: Summarize customer attributes
    summarize_customer_attributes(df)

    # Step 2: Cluster customers based on CLTV score
    clustered_customers = cluster_customers(df, num_clusters=3)
    
    # Step 3: Show the result of the clustering
    print("\nClustered customer data:")
    print(clustered_customers.head())
