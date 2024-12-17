import os
from tkinter import NO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kstest
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif



# column headers
column_names = [
    "PolicyNumber", "PolicyDuration", "TimeToEffective", "PurchaseMonth", "EffectiveMonth",
    "NumberOfInsured", "MedicalUnderwritingFlag", "StandardCustom", "SingleMulti", "ProductID",
    "ProductName", "ProductGroup", "Category", "CoverageCode", "CoverageName", "InsuredAge",
    "InsuredProvince", "Destination", "UnderwriterName", "SellerName", "SalesChannel",
    "TotalEarnedPremium", "ClaimsFrequency", "TotalClaimAmount", "AverageClaimAmount",
    "TotalChargedAmountCAD", "TotalPaidAmountCAD", "TotalPPONetworkFeeCAD", "LossRatio",
    "IsProfitable"
]



# load the dataset
file_path = r'C:\Users\abadianov\Desktop\Projects\ML\dataset.csv'
data = pd.read_csv(file_path, names=column_names, header=None)
# print(data.head())
# print(data.info())
# print(data.dtypes)




# Fix the dataset
# Remove rows with negative values
data = data[(data['TimeToEffective'] >= 0) & (data['TotalEarnedPremium'] >= 0) & (data['TotalClaimAmount'] >= 0)]

# Drop irrelevant features
data = data.drop(columns=['TotalPPONetworkFeeCAD', 'PolicyNumber', 'PolicyDuration',
                          'TotalChargedAmountCAD', 'TotalEarnedPremium', 'ClaimsFrequency',
                          'TotalClaimAmount','AverageClaimAmount', 'TotalPaidAmountCAD', 'SellerName',
                          'UnderwriterName', 'ProductID', 'CoverageCode'])

# Fix Data Types
data['MedicalUnderwritingFlag'] = data['MedicalUnderwritingFlag'].astype(bool)
data['IsProfitable'] = data['IsProfitable'].astype(bool)

# Find missing values
missing_values = data.isnull().sum().sort_values(ascending=False)
missing_percent = (missing_values / data.shape[0]) * 100
missing_table = pd.concat([missing_values, missing_percent], axis=1, keys=['Missing Values', 'Percentage'])
# print("Missing Values per Column:")
# print(missing_table)

data['Destination'] = data['Destination'].fillna('Unknown')
most_common_province = data['InsuredProvince'].mode()[0]
data['InsuredProvince'] = data['InsuredProvince'].fillna(most_common_province)
data = data.dropna()





# Analyze the dataset
# Num and cat cols
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object', 'bool']).columns.tolist()

# Statistical summary
stats_summary = data.describe().transpose()
stats_summary['skewness'] = data[numerical_cols].skew()
stats_summary['kurtosis'] = data[numerical_cols].kurtosis()
print(stats_summary)

# Fixing skewness and kurtosis
skewed_features = [
    'TimeToEffective', 'NumberOfInsured'
]

# Apply Yeo-Johnson transformation
pt = PowerTransformer(method='yeo-johnson')
data[skewed_features] = pt.fit_transform(data[skewed_features])

# Recalculate skewness and kurtosis
stats_summary_transformed = data[skewed_features].describe().transpose()
stats_summary_transformed['skewness'] = data[skewed_features].skew()
stats_summary_transformed['kurtosis'] = data[skewed_features].kurtosis()
print(stats_summary_transformed[['skewness', 'kurtosis']])

# Capping outliers
for col in skewed_features:
    lower_cap = data[col].quantile(0.01)
    upper_cap = data[col].quantile(0.99)
    data[col] = data[col].clip(lower=lower_cap, upper=upper_cap)

# Z-scores method for outliers
z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
threshold = 3
outlier_rows = (z_scores > threshold).any(axis=1)
print(f"Number of outlier observations: {outlier_rows.sum()}")

# IQR method for outliers
Q1 = data[numerical_cols].quantile(0.25)
Q3 = data[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
outlier_condition = ((data[numerical_cols] < (Q1 - 1.5 * IQR)) | (data[numerical_cols] > (Q3 + 1.5 * IQR)))
outlier_rows_iqr = outlier_condition.any(axis=1)
print(f"Number of outlier observations (IQR method): {outlier_rows_iqr.sum()}")

# Kolmogorov-Smirnov Test for normality analysis - completed and showed that none of features is normally distributed

# Correlation analysis
corr_matrix = data[numerical_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
threshold = 0.8
high_corr_features = [column for column in upper.columns if any(upper[column] > threshold)]

print("Highly correlated features (correlation > 0.8):")
print(high_corr_features)

for feature in high_corr_features:
    correlated_with = upper.index[upper[feature] > threshold].tolist()
    print(f"{feature} is highly correlated with: {correlated_with}")

# Binning 'InsuredAge' into categories
age_bins = [0, 17, 25, 35, 45, 55, 60, 65, 70, 75, np.inf]
age_labels = ['0-17', '18-25', '26-35', '36-45', '46-55', '56-60', '61-65', '66-70', '71-75', '76+']
data['AgeGroup'] = pd.cut(data['InsuredAge'], bins=age_bins, labels=age_labels, right=False)
data = data.drop(columns=['InsuredAge'])
data['AgeGroup'] = data['AgeGroup'].astype('object')





# Categorical cols encoding and standartization
# Keep a copy of the original data before encoding
data_original = data.copy()

# Convert boolean columns to integers
boolean_cols = data.select_dtypes(include='bool').columns.tolist()
for col in boolean_cols:
    data[col] = data[col].astype(int)

# Re-identify categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# Separate low and high cardinality columns before encoding
low_cardinality_cols = [col for col in categorical_cols if data[col].nunique() <= 10]
high_cardinality_cols = [col for col in categorical_cols if data[col].nunique() > 10]

# Apply One-Hot Encoding to low cardinality columns
data = pd.get_dummies(data, columns=low_cardinality_cols, dtype=int)

# Apply Frequency Encoding to high cardinality columns
frequency_encoding_maps = {}  # Dictionary to store encoding maps

for col in high_cardinality_cols:
    frequency_encoding = data[col].value_counts() / len(data)
    data[col] = data[col].map(frequency_encoding)
    # Store the mapping
    frequency_encoding_maps[col] = frequency_encoding

# Re-identify numerical columns after transformations
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Exclude target variables
if 'IsProfitable' in numerical_cols:
    numerical_cols.remove('IsProfitable')  # Exclude target variable
if 'LossRatio' in numerical_cols:
    numerical_cols.remove('LossRatio')     # Exclude target variable

# Initialize the scaler
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Verify that all columns are numeric
remaining_categorical_cols = data.select_dtypes(include=['object', 'bool']).columns.tolist()
if remaining_categorical_cols:
    print("Remaining non-numeric columns:", remaining_categorical_cols)
else:
    print("All columns are now numeric.")

# Check for missing values after encoding
missing_values = data.isnull().sum()
print("Missing values after encoding:")
print(missing_values[missing_values > 0])





# Finding Optimal K
# X = data.drop(columns=['IsProfitable', 'LossRatio'])


# k_values = range(2, 11) 

# inertia = []
# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X)
#     inertia.append(kmeans.inertia_)

# plt.figure(figsize=(8, 4))
# plt.plot(k_values, inertia, 'bo-')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method for Optimal k')
# plt.show()


# silhouette_scores = []
# sample_X = X.iloc[np.random.choice(X.shape[0], 60000, replace=False), :]

# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(sample_X)
#     score = silhouette_score(sample_X, labels)
#     silhouette_scores.append(score)

# plt.figure(figsize=(8, 4))
# plt.plot(k_values, silhouette_scores, 'bo-')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Analysis for Optimal k')
# plt.show()


# davies_bouldin_scores = []

# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(X)
#     score = davies_bouldin_score(X, labels)
#     davies_bouldin_scores.append(score)

# plt.figure(figsize=(8, 4))
# plt.plot(k_values, davies_bouldin_scores, 'bo-')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Davies-Bouldin Index')
# plt.title('Davies-Bouldin Analysis for Optimal k')
# plt.show()


# calinski_harabasz_scores = []

# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(X)
#     score = calinski_harabasz_score(X, labels)
#     calinski_harabasz_scores.append(score)

# plt.figure(figsize=(8, 4))
# plt.plot(k_values, calinski_harabasz_scores, 'bo-')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Calinski-Harabasz Score')
# plt.title('Calinski-Harabasz Analysis for Optimal k')
# plt.show()





# Feature Importance Analysis for Feature Selection

# Exclude financial variables from clustering
excluded_features = ['LossRatio', 'IsProfitable']

# Prepare feature matrix X
X = data.drop(columns=excluded_features + ['Cluster'], errors='ignore')
selected_features = X.columns.tolist()
print("Selected Features for Feature Importance Analysis:", selected_features)

# Initialize the scaler (already done earlier)
X_scaled = X  # Data is already scaled

# Perform initial clustering to obtain cluster labels
kmeans_initial = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans_initial.fit_predict(X_scaled)

# Add cluster labels to the data
data['Cluster_Initial'] = cluster_labels

# Prepare the feature matrix and target variable for Random Forest
X_rf = X_scaled
y_rf = data['Cluster_Initial']

# Train Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_classifier.fit(X_rf, y_rf)

# Get feature importances
importances = rf_classifier.feature_importances_
feature_names = selected_features

# Create a DataFrame for easy visualization
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort features by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Display feature importances in text format
print("\nFeature importances based on Random Forest (overall):")
for index, row in feature_importances.iterrows():
    feature = row['Feature']
    importance = row['Importance']
    print(f"Feature: {feature}, Importance: {importance:.6f}")

# Decide on a threshold to remove low-importance features
importance_threshold = 0.01  # You can adjust this threshold
low_importance_features = feature_importances[feature_importances['Importance'] < importance_threshold]['Feature'].tolist()

print("\nFeatures suggested for removal (importance < {:.2f}):".format(importance_threshold))
for feature in low_importance_features:
    print(f"- {feature}")

# Optionally, remove low-importance features from X
# X_selected = X_rf.drop(columns=low_importance_features)





# Algorithms

# Exclude financial variables from clustering
excluded_features = ['LossRatio', 'IsProfitable']

print(data.columns)

# Prepare feature matrix X
X = data.drop(columns=excluded_features + ['Cluster'] + ['Cluster_Initial'], errors='ignore')
selected_features = X.columns.tolist()
print("Selected Features for Clustering:", selected_features)

# Initialize the scaler (already done earlier)
X_scaled = X  # Data is already scaled

def cluster_and_analyze(X, algorithm, algorithm_name):
    print(f"\n--- Clustering with {algorithm_name} ---")
    
    # Fit the clustering algorithm
    cluster_labels = algorithm.fit_predict(X)
    
    # Add cluster labels to data
    data['Cluster'] = cluster_labels
    
    # Remove noise points for DBSCAN
    data_filtered = data[data['Cluster'] != -1] if algorithm_name == 'DBSCAN' else data.copy()
    
    # Cluster sizes
    cluster_sizes = data_filtered['Cluster'].value_counts().sort_index()
    print("Cluster Sizes:")
    print(cluster_sizes)
    
    # Analyze cluster characteristics
    cluster_profiles = data_filtered.groupby('Cluster').mean()
    print("Cluster Profiles:")
    print(cluster_profiles)
    
    # Analyze feature importance for each cluster
    analyze_feature_importance_per_cluster(data_filtered, selected_features)
    
    # Calculate and display average LossRatio by cluster
    profitability_by_cluster = data_filtered.groupby('Cluster')['LossRatio'].mean()
    print("Average LossRatio by Cluster:")
    print(profitability_by_cluster)
    
    return data_filtered

def analyze_feature_importance_per_cluster(data, features):
    from sklearn.ensemble import RandomForestClassifier
    
    print("\nFeature importances for each cluster based on Random Forest:")
    
    for cluster_label in sorted(data['Cluster'].unique()):
        print(f"\n--- Cluster {cluster_label} ---")
        
        # Create binary target variable for one-vs-rest classification
        data['Target'] = (data['Cluster'] == cluster_label).astype(int)
        
        # Prepare the feature matrix and target variable
        X_rf = data[features]
        y_rf = data['Target']
        
        # Initialize and fit the Random Forest Classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_classifier.fit(X_rf, y_rf)
        
        # Get feature importances
        importances = rf_classifier.feature_importances_
        feature_names = features
        
        # Create a DataFrame for easy visualization
        feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        
        # Sort features by importance
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
        
        # Display top features for the cluster
        print("Top features for this cluster:")
        print(feature_importances.head(10))
        
        # Extract data for the current cluster
        cluster_data = data[data['Cluster'] == cluster_label]
        
        # Interpret the top features
        interpret_top_features(feature_importances, cluster_data, cluster_label)
       
        # Clean up the Target column
        data.drop(columns=['Target'], inplace=True)


def interpret_top_features(feature_importances, cluster_data, cluster_label):
    print("\nInterpreting Top Features:")
    for index, row in feature_importances.head(5).iterrows():
        feature = row['Feature']
        importance = row['Importance']
        
        print(f"\nFeature: {feature} (Importance: {importance:.4f})")
        
        # Check if the feature was frequency encoded
        if feature in high_cardinality_cols:
            # Get the original categories from the cluster data
            original_categories = data_original.loc[cluster_data.index, feature]
            category_counts = original_categories.value_counts(normalize=True).head(5)
            print(f"Top Categories and Frequencies in Cluster {cluster_label}:")
            print(category_counts)
        elif feature in low_cardinality_cols:
            # For one-hot encoded features
            base_feature = feature
            categories = cluster_data[feature].value_counts(normalize=True).head(5)
            print(f"Category Frequencies in Cluster {cluster_label}:")
            print(categories)
        else:
            # Numerical feature
            print("Numerical Feature")


# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
data_kmeans = cluster_and_analyze(X_scaled, kmeans, 'K-Means')


# Perform PCA with 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with the PCA components and cluster labels
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Cluster'] = data_kmeans['Cluster'].values  # Ensure cluster labels are aligned

# Plot the PCA components with clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=50)
plt.title('PCA Visualization of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# Gaussian Mixture Models
# gmm = GaussianMixture(n_components=10, random_state=42)
# gmm.fit(X_scaled)
# cluster_labels = gmm.predict(X_scaled)
# data['Cluster'] = cluster_labels
# data_gmm = data.copy()
# cluster_and_analyze(X_scaled, gmm, 'GMM')
