import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
import hdbscan


# Load the dataset
file_path = r'C:\Users\abadianov\Desktop\Projects\ML\datasetv2_dest_regionv3.csv'
data = pd.read_csv(file_path)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Filter data for 'EHM' ProductGroup
data = data[data['ProductGroup'] == 'EHM']
data = data[data['DestinationRegion'] != 'Other']
print(data['DestinationRegion'].value_counts())
print(data['PartnerSegment'].value_counts())






# Fix the dataset
# Remove rows with negative values
data = data[(data['TimeToEffective'] >= 0) & (data['TotalEarnedPremium'] >= 0) & (data['TotalClaimAmount'] >= 0)]

# Drop financial features
data = data.drop(columns=[
    'PolicyNumber', 'TotalPPONetworkFeeCAD', 'TotalChargedAmountCAD', 'TotalEarnedPremium',
    'TotalClaimAmount', 'AverageClaimAmount', 'TotalPaidAmountCAD', 'IsProfitable'
])

# Fix Data Types
data['MedicalUnderwritingFlag'] = data['MedicalUnderwritingFlag'].astype(bool)

# Handle missing values
data['Destination'] = data['Destination'].fillna('Unknown')
most_common_province = data['InsuredProvince'].mode()[0]
data['InsuredProvince'] = data['InsuredProvince'].fillna(most_common_province)
data = data.dropna()

# Fix 'PartnerSegment'
data['PartnerSegment'] = data['PartnerSegment'].replace('Broker', 'BROKER')
low_value_segments = ['OTHER', 'B2C', 'Group', 'Service']
data['PartnerSegment'] = data['PartnerSegment'].replace(low_value_segments, 'Other')


# Set to Broker
data = data[data['PartnerSegment'] == 'BROKER']
data = data.drop(columns=['PartnerSegment'])

# # Set to All Others
# data = data[data['PartnerSegment'] == 'Airlines']


# Drop SPECIFIC TO EHM
data = data.drop(columns=[
    'ProductGroup', 'ProductID', 'ProductName', 'UnderwriterName', 'CoverageCode',
    'PolicyDuration', 'TimeToEffective', 'PurchaseMonth', 'EffectiveMonth',
    'MedicalUnderwritingFlag', 'StandardCustom', 'SingleMulti', 'Category',
    'InsuredProvince', 'SellerName', 'SalesChannel', 'CoverageName', 'NumberOfInsured',
    'ClaimsFrequency', 'MappedDestination', 'Destination', 'SimplifiedDestination'
])






# Analyze the dataset
# Identify numerical and categorical columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object', 'bool']).columns.tolist()

# Statistical summary
stats_summary = data.describe().transpose()
stats_summary['skewness'] = data[numerical_cols].skew()
stats_summary['kurtosis'] = data[numerical_cols].kurtosis()
print(stats_summary)

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

# Correlation analysis
corr_matrix = data[numerical_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
threshold_corr = 0.8
high_corr_features = [column for column in upper.columns if any(upper[column] > threshold_corr)]

print("Highly correlated features (correlation > 0.8):")
print(high_corr_features)

for feature in high_corr_features:
    correlated_with = upper.index[upper[feature] > threshold_corr].tolist()
    print(f"{feature} is highly correlated with: {correlated_with}")

# Categorical columns encoding and standardization
# Binning 'InsuredAge' into categories
age_bins = [0, 17, 25, 35, 45, 55, 60, 65, 70, 75, np.inf]
age_labels = ['0-17', '18-25', '26-35', '36-45', '46-55', '56-60', '61-65', '66-70', '71-75', '76+']
data['AgeGroup'] = pd.cut(data['InsuredAge'], bins=age_bins, labels=age_labels, right=False)
data = data.drop(columns=['InsuredAge'])
data['AgeGroup'] = data['AgeGroup'].astype('object')

# Keep a copy of the original data before encoding
data_original = data.copy()

# Map 'AgeGroup' to ordinal numerical codes
age_group_order = {
    '0-17': 1,
    '18-25': 2,
    '26-35': 3,
    '36-45': 4,
    '46-55': 5,
    '56-60': 6,
    '61-65': 7,
    '66-70': 8,
    '71-75': 9,
    '76+': 10
}
data['AgeGroup'] = data['AgeGroup'].map(age_group_order)

# Create 'AgeGroup_Label' for data and data_original
data['AgeGroup_Label'] = data['AgeGroup'].map(lambda x: age_labels[x-1])
data_original['AgeGroup_Label'] = data_original['AgeGroup']

# Re-identify categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# Separate low and high cardinality columns before encoding
low_cardinality_cols = [col for col in categorical_cols if data[col].nunique() <= 2]
high_cardinality_cols = [col for col in categorical_cols if data[col].nunique() > 2]

# Apply One-Hot Encoding to low cardinality columns
data = pd.get_dummies(data, columns=low_cardinality_cols, dtype=int)

# Apply Frequency Encoding to high cardinality columns
frequency_encoding_maps = {}  # Dictionary to store encoding maps

for col in high_cardinality_cols:
    frequency_encoding = data[col].value_counts() / len(data)
    data[col] = data[col].map(frequency_encoding)
    frequency_encoding_maps[col] = frequency_encoding

# Re-identify numerical columns after transformations
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Exclude target variables
for target_col in ['IsProfitable', 'LossRatio']:
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)

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







# # Group by 'DestinationRegion' and 'AgeGroup_Label' and calculate the mean of 'LossRatio'
# grouped_data = data_original.groupby(['DestinationRegion', 'AgeGroup_Label'])['LossRatio'].mean().reset_index()
# pivot_table = grouped_data.pivot(index='AgeGroup_Label', columns='DestinationRegion', values='LossRatio')
# print(pivot_table)
# output_file_path = r'C:\Users\abadianov\Desktop\Projects\ML\pivot_table_airlines.xlsx'
# pivot_table.to_excel(output_file_path, sheet_name='PivotTable', index=True)
# print(f"The pivot table has been saved as '{output_file_path}'.")


# Group by 'DestinationRegion' and 'AgeGroup_Label' and calculate the count of rows
grouped_data_counts = data_original.groupby(['DestinationRegion', 'AgeGroup_Label']).size().reset_index(name='RowCount')
pivot_table_counts = grouped_data_counts.pivot(index='AgeGroup_Label', columns='DestinationRegion', values='RowCount')
print(pivot_table_counts)
output_file_path_counts = r'C:\Users\abadianov\Desktop\Projects\ML\pivot_table_counts_broker.xlsx'
pivot_table_counts.to_excel(output_file_path_counts, sheet_name='RowCounts', index=True)
print(f"The pivot table with row counts has been saved as '{output_file_path_counts}'.")




# Algorithms
# Exclude variables from clustering
excluded_features = ['LossRatio', 'AgeGroup_Label']  # Exclude the label column

# Prepare feature matrix X
X = data.drop(columns=excluded_features + ['Cluster'], errors='ignore')
selected_features = X.columns.tolist()
print("Selected Features for Clustering:", selected_features)

X_scaled = X

def cluster_and_analyze(X, algorithm, algorithm_name, data_original, selected_features, high_cardinality_cols, low_cardinality_cols):
    print(f"\n--- Clustering with {algorithm_name} ---")
    # Fit the clustering algorithm
    if algorithm_name == 'HDBSCAN':
        cluster_labels = algorithm.fit_predict(X)
    elif algorithm_name == 'Gaussian Mixture':
        algorithm.fit(X)
        cluster_labels = algorithm.predict(X)
    else:
        cluster_labels = algorithm.fit_predict(X)
    
    data['Cluster'] = cluster_labels
    data_original['Cluster'] = cluster_labels  # Add Cluster labels to original data
    data_filtered = data[data['Cluster'] != -1] if algorithm_name == 'HDBSCAN' else data.copy()
    data_original_filtered = data_original.loc[data_filtered.index]  # Filter original data accordingly

    # Cluster sizes
    cluster_sizes = data_filtered['Cluster'].value_counts().sort_index()
    print("\nCluster Sizes:")
    print(cluster_sizes)

    # Create cluster profiles (includes feature importance)
    create_cluster_profiles(data_filtered, data_original_filtered, selected_features, high_cardinality_cols, low_cardinality_cols)

    # Calculate and display average LossRatio by cluster
    profitability_by_cluster = data_filtered.groupby('Cluster')['LossRatio'].mean()
    print("\nAverage LossRatio by Cluster:")
    print(profitability_by_cluster)

    return data_filtered

def create_cluster_profiles(data, data_original, features, high_cardinality_cols, low_cardinality_cols):
    print("\nCluster Profiles:")

    for cluster_label in sorted(data['Cluster'].unique()):
        print(f"\n--- Cluster {cluster_label} ---")
        cluster_data = data[data['Cluster'] == cluster_label]
        cluster_data_original = data_original.loc[cluster_data.index]  # Ensure alignment of indices

        # Compute feature importances for this cluster
        # Random Forest Classifier
        data['Target'] = (data['Cluster'] == cluster_label).astype(int)
        X_rf = data[features]
        y_rf = data['Target']
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_classifier.fit(X_rf, y_rf)

        # Top features calculation
        importances = rf_classifier.feature_importances_
        feature_names = features
        feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
        top_features = feature_importances.head(3)

        # Include the importance numbers in the cluster profile
        print("\nTop Features for This Cluster:")
        for idx, row in top_features.iterrows():
            print(f"Feature: {row['Feature']}, Importance: {row['Importance']:.6f}")

        # Now output the top categories for the top features
        for index, row in top_features.iterrows():
            feature = row['Feature']
            importance = row['Importance']
            print(f"\nFeature: {feature} (Importance: {importance:.4f})")

            if feature in high_cardinality_cols or feature in low_cardinality_cols:
                if feature in cluster_data_original.columns:
                    original_feature_data = cluster_data_original[feature]
                    top_categories = original_feature_data.value_counts(normalize=True).head(5).mul(100).round(2)
                    print(f"Top Categories and Frequencies in Cluster {cluster_label}:")
                    for category, percentage in top_categories.items():
                        print(f"{category}: {percentage}%")
                else:
                    print(f"Feature {feature} not found in original data.")
            else:
                print("Numerical Feature")

        # Provide detailed interpretation of AgeGroup
        if 'AgeGroup_Label' in cluster_data_original.columns:
            age_distribution = (
                cluster_data_original['AgeGroup_Label']
                .value_counts(normalize=True)
                .mul(100)
                .round(2)
            )
            print("\nAgeGroup Distribution:")
            for age_group, percentage in age_distribution.items():
                print(f"{age_group}: {percentage}%")
        else:
            print("AgeGroup_Label not found in original data.")

    data.drop(columns=['Target'], inplace=True)

# K-Means Clustering
algorithm_name = f'K-Means (k={5})'
algorithm = KMeans(n_clusters=5, random_state=42)
data_clustered = cluster_and_analyze(X_scaled, algorithm, algorithm_name, data_original, selected_features, high_cardinality_cols, low_cardinality_cols)

# # Loop over k_values and run K-Means clustering
# k_values = range(2, 10)
# for k in k_values:
#     algorithm_name = f'K-Means (k={k})'
#     algorithm = KMeans(n_clusters=k, random_state=42)
#     data_clustered = cluster_and_analyze(X_scaled, algorithm, algorithm_name, data_original, selected_features, high_cardinality_cols, low_cardinality_cols)