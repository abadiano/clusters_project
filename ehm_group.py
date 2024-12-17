import os
from tkinter import NO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# SET TO PRODUCT
data = data[data['ProductName'] == 'Emergency Medical Plan']



# Quick insight
age_bins = [0, 17, 25, 35, 45, 55, 60, 65, np.inf]
age_labels = ['0-17', '18-25', '26-35', '36-45', '46-55', '56-60', '61-65', '66+']
data['AgeGroup'] = pd.cut(data['InsuredAge'], bins=age_bins, labels=age_labels, right=False)
grouped_data_with_age = data.groupby(['AgeGroup']).agg(
    AverageLossRatio=('LossRatio', 'mean'),
    Count=('LossRatio', 'size')
).reset_index()

grouped_data_with_age.columns = ['AgeGroup', 'AverageLossRatio', 'Count']
print(grouped_data_with_age)

