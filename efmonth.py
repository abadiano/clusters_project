import pandas as pd
import numpy as np

# Load the dataset
file_path = r'C:\Users\abadianov\Desktop\Projects\ML\datasetv2_dest_regionv3.csv'
df = pd.read_csv(file_path)

# Filter for EHM and United States
df = df[df['ProductGroup'] == 'EHM']
df = df[df['DestinationRegion'] == 'United States']

# Fix 'PartnerSegment'
df['PartnerSegment'] = df['PartnerSegment'].replace('Broker', 'BROKER')
low_value_segments = ['OTHER', 'B2C', 'Group', 'Service']
df['PartnerSegment'] = df['PartnerSegment'].replace(low_value_segments, 'Other')

# Set to Broker
df = df[df['PartnerSegment'] == 'BROKER']
df = df.drop(columns=['PartnerSegment'])

# Remove rows with negative values
df = df[(df['TimeToEffective'] >= 0) & (df['TotalEarnedPremium'] >= 0) & (df['TotalClaimAmount'] >= 0)]

print(df.shape)


df = df[df['ProductName'] == 'Emergency Medical Plan']


# Age binning
age_bins = [0, 17, 25, 35, 45, 55, 60, 65, 70, 75, np.inf]
age_labels = ['0-17', '18-22', '23-32', '33-45', '46-55', '56-60', '61-65', '66-70', '71-75', '76+']
df['AgeGroup'] = pd.cut(df['InsuredAge'], bins=age_bins, labels=age_labels, right=False)

# # Filter the dataset for the AgeGroup '26-35'
# df = df[df['AgeGroup'] == '26-35']


# Filter the dataset for the AgeGroup '0-17' and '26-35'
# df_filtered = df[df['AgeGroup'].isin(['18-25', '26-35', '36-45'])]
# df_filtered = df[df['AgeGroup'].isin(['23-32'])]
df_filtered = df

print(df.nunique())


# # Drop financial features
# df = df.drop(columns=['PolicyNumber', 'TotalPPONetworkFeeCAD',
#                           'TotalChargedAmountCAD', 'TotalEarnedPremium',
#                           'TotalClaimAmount', 'AverageClaimAmount', 'TotalPaidAmountCAD', 'IsProfitable'])

# df = df.drop(columns=['MedicalUnderwritingFlag',  
# 'StandardCustom',              
# 'SingleMulti',                
# 'ProductID',                    
# 'ProductName',                
# 'ProductGroup',                 
# 'Category' ,                    
# 'CoverageCode' ,               
# 'CoverageName', 'MappedDestination',         
# 'SimplifiedDestination',      
# 'DestinationRegion',         
# 'Destination',              
# 'UnderwriterName','ClaimsFrequency',
# 'AgeGroup', 'InsuredProvince', 'SalesChannel', 'PurchaseMonth', 
# 'PolicyDuration',     
# 'TimeToEffective',    
# 'NumberOfInsured'])


# print(df.nunique())


# # Calculate average LossRatio and row counts for all rows
# average_loss_ratio = df_filtered['LossRatio'].mean()
# total_rows = df_filtered['LossRatio'].count()

# # Map LossRatio and row counts to selected columns
# loss_ratio_mapping = {
#     "InsuredAge": df_filtered.groupby("InsuredAge").agg(
#         AvgLossRatio=("LossRatio", "mean"),
#         RowCount=("LossRatio", "size")
#     ),
#     "EffectiveMonth": df_filtered.groupby("EffectiveMonth").agg(
#         AvgLossRatio=("LossRatio", "mean"),
#         RowCount=("LossRatio", "size")
#     )
# }

# # Convert mapping results to DataFrames for user visualization
# loss_ratio_dfs = {
#     key: value.reset_index() for key, value in loss_ratio_mapping.items()
# }

# # Display the average LossRatio and row count
# average_loss_ratio_df = pd.DataFrame({
#     "AvgLossRatio": [average_loss_ratio],
#     "RowCount": [total_rows]
# })
# print("Average Loss Ratio and Row Count:")
# print(average_loss_ratio_df)

# # Display the mappings
# for name, dataframe in loss_ratio_dfs.items():
#     print(f"Loss Ratio by {name}:")
#     print(dataframe)