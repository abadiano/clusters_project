import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# column headers
column_names = [
    "PolicyNumber", "PolicyDuration", "TimeToEffective", "PurchaseMonth", "EffectiveMonth",
    "NumberOfInsured", "MedicalUnderwritingFlag", "StandardCustom", "SingleMulti", "ProductID",
    "ProductName", "ProductGroup", "Category", "CoverageCode", "CoverageName", "InsuredAge",
    "InsuredProvince", "Destination", "UnderwriterName", "SellerName", "SalesChannel",
    "TotalEarnedPremium", "PartnerSegment", "ClaimsFrequency", "TotalClaimAmount", "AverageClaimAmount",
    "TotalChargedAmountCAD", "TotalPaidAmountCAD", "TotalPPONetworkFeeCAD", "LossRatio",
    "IsProfitable"
]

# Load the main dataset
file_path = r'C:\Users\abadianov\Desktop\Projects\ML\Datasets\datasetv2.csv'
data = pd.read_csv(file_path, names=column_names, header=None)
print("Initial Shape: ", data.shape)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)




# Load Index Data
file_path1 = r'C:\Users\abadianov\Desktop\Projects\ML\CountryIndex\prices.csv'
file_path2 = r'C:\Users\abadianov\Desktop\Projects\ML\CountryIndex\system.csv'
file_path3 = r'C:\Users\abadianov\Desktop\Projects\ML\CountryIndex\safety.csv'

prices = pd.read_csv(file_path1)
system = pd.read_csv(file_path2)
safety = pd.read_csv(file_path3)

# Map Safety: level is 1 to 4 -> convert to score
def map_safety_to_score(level):
    mapping = {1:0, 2:3, 3:7, 4:10}
    return mapping.get(level, 5) # default if unexpected

safety['Safety_Score'] = safety['Risk'].apply(map_safety_to_score)

# Health system: (1 - WHOIndex) * 10
system['HealthSystem_Score'] = (1 - system['TheWorldWHOIndex2000']) * 10

# Normalize price to 0-10 using all prices data
min_price = prices['Healthcare prices'].min()
max_price = prices['Healthcare prices'].max()

def normalize_price(price):
    return ((price - min_price) / (max_price - min_price)) * 10

prices['Price_Score'] = prices['Healthcare prices'].apply(normalize_price)

# Create dictionaries for lookup using all countries
safety_dict = dict(zip(safety['Country'], safety['Safety_Score']))
system_dict = dict(zip(system['Country'], system['HealthSystem_Score']))
price_dict = dict(zip(prices['Country'], prices['Price_Score']))

# Basic cleaning: Map Destinations to Countries if possible (Assuming you have city/airport mapping)
city_country_mapping = pd.read_csv(r'C:\Users\abadianov\Desktop\Projects\ML\city_to_country_mapping.csv')
city_country_dict = dict(zip(city_country_mapping["name"], city_country_mapping["Country"]))

airports = pd.read_csv(r'C:\Users\abadianov\Desktop\Projects\ML\airports_to_code_mapping.csv', usecols=["IATA", "Country"])
airport_to_country_dict = dict(zip(airports["IATA"], airports["Country"]))

data['MappedDestination'] = (
    data['Destination']
    .map(city_country_dict)
    .fillna(data['Destination'].map(airport_to_country_dict))
    .fillna(data['Destination'])
)

# Simplify and standardize destination
data['SimplifiedDestination'] = data['MappedDestination'].str.title().fillna('Other')
data['SimplifiedDestination'] = data['SimplifiedDestination'].apply(lambda x: 'Other' if any(char.isdigit() for char in x) else x)

# Functions to map scores to the dataset
def get_safety_score(country): return safety_dict.get(country, None)
def get_system_score(country): return system_dict.get(country, None)
def get_price_score(country): return price_dict.get(country, None)

data['Safety_Score'] = data['SimplifiedDestination'].apply(get_safety_score)
data['System_Score'] = data['SimplifiedDestination'].apply(get_system_score)
data['Price_Score'] = data['SimplifiedDestination'].apply(get_price_score)

# Filter for EHM ProductGroup if needed
data = data[data['ProductGroup'] == 'EHM']
print("EHM Shape: ", data.shape)

# Keep only rows with a LossRatio
data = data.dropna(subset=['LossRatio'])
print("LossRatio Shape: ", data.shape)

data_safety = data.dropna(subset=['Safety_Score'])
print("Safety_Score Shape: ", data_safety.shape)

data_system = data.dropna(subset=['System_Score'])
print("System_Score Shape: ", data_system.shape)

data_price = data.dropna(subset=['Price_Score'])
print("Price_Score Shape: ", data_price.shape)

# Define categories based on previous reasoning:
# Safety_Score: Low (0), Medium (3), High (≥7)
def categorize_safety(score):
    if score == 0:
        return 'Low'
    elif score == 3:
        return 'Medium'
    else:
        return 'High'  # covers 7, 10

data_safety['Safety_Category'] = data_safety['Safety_Score'].apply(categorize_safety)

# System_Score: Low (≤1.5), Medium (1.5 < x ≤3.0), High (>3.0)
def categorize_system(score):
    if score <= 1.5:
        return 'Low'
    elif score <= 3.0:
        return 'Medium'
    else:
        return 'High'

data_system['System_Category'] = data_system['System_Score'].apply(categorize_system)

# Price_Score: Low (≤1.5), Medium (1.5 < x ≤5.0), High (>5.0)
def categorize_price(score):
    if score <= 1.5:
        return 'Low'
    elif score <= 5.0:
        return 'Medium'
    else:
        return 'High'

data_price['Price_Category'] = data_price['Price_Score'].apply(categorize_price)

# Group by categories and compute mean LossRatio and row count
safety_stats = data_safety.groupby('Safety_Category')['LossRatio'].agg(['mean', 'count']).reset_index()
safety_stats.rename(columns={'mean': 'Mean_LossRatio', 'count': 'Row_Count'}, inplace=True)
print("\n=== Safety_Category Statistics ===")
print(safety_stats)

system_stats = data_system.groupby('System_Category')['LossRatio'].agg(['mean', 'count']).reset_index()
system_stats.rename(columns={'mean': 'Mean_LossRatio', 'count': 'Row_Count'}, inplace=True)
print("\n=== System_Category Statistics ===")
print(system_stats)

price_stats = data_price.groupby('Price_Category')['LossRatio'].agg(['mean', 'count']).reset_index()
price_stats.rename(columns={'mean': 'Mean_LossRatio', 'count': 'Row_Count'}, inplace=True)
print("\n=== Price_Category Statistics ===")
print(price_stats)






# Define age bins and labels
age_bins = [0, 17, 25, 35, 45, 55, 60, 65, 70, 75, np.inf]
age_labels = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-59', '60-64', '65-69', '70-74', '75+']

# Add AgeGroup column to each subset
data_safety['AgeGroup'] = pd.cut(data_safety['InsuredAge'], bins=age_bins, labels=age_labels, right=False)
data_system['AgeGroup'] = pd.cut(data_system['InsuredAge'], bins=age_bins, labels=age_labels, right=False)
data_price['AgeGroup'] = pd.cut(data_price['InsuredAge'], bins=age_bins, labels=age_labels, right=False)

# Safety analysis by AgeGroup and Safety_Category
safety_mean = data_safety.groupby(['AgeGroup', 'Safety_Category'])['LossRatio'].mean().unstack(level=1).fillna(0)
safety_count = data_safety.groupby(['AgeGroup', 'Safety_Category'])['LossRatio'].count().unstack(level=1).fillna(0)

safety_age_stats = pd.concat([safety_mean.add_prefix('Mean_LossRatio_'), safety_count.add_prefix('Row_Count_')], axis=1)

print("\n=== Safety_Category by AgeGroup ===")
print(safety_age_stats)

# System analysis by AgeGroup and System_Category
system_mean = data_system.groupby(['AgeGroup', 'System_Category'])['LossRatio'].mean().unstack(level=1).fillna(0)
system_count = data_system.groupby(['AgeGroup', 'System_Category'])['LossRatio'].count().unstack(level=1).fillna(0)

system_age_stats = pd.concat([system_mean.add_prefix('Mean_LossRatio_'), system_count.add_prefix('Row_Count_')], axis=1)

print("\n=== System_Category by AgeGroup ===")
print(system_age_stats)

# Price analysis by AgeGroup and Price_Category
price_mean = data_price.groupby(['AgeGroup', 'Price_Category'])['LossRatio'].mean().unstack(level=1).fillna(0)
price_count = data_price.groupby(['AgeGroup', 'Price_Category'])['LossRatio'].count().unstack(level=1).fillna(0)

price_age_stats = pd.concat([price_mean.add_prefix('Mean_LossRatio_'), price_count.add_prefix('Row_Count_')], axis=1)

print("\n=== Price_Category by AgeGroup ===")
print(price_age_stats)