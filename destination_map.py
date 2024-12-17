import pandas as pd
import os

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

# load the dataset
file_path = r'C:\Users\abadianov\Desktop\Projects\ML\Datasets\datasetv2.csv'
data = pd.read_csv(file_path, names=column_names, header=None)









# Load Index Data
file_path1 = r'C:\Users\abadianov\Desktop\Projects\ML\CountryIndex\prices.csv'
file_path2 = r'C:\Users\abadianov\Desktop\Projects\ML\CountryIndex\system.csv'
file_path3 = r'C:\Users\abadianov\Desktop\Projects\ML\CountryIndex\safety.csv'
prices = pd.read_csv(file_path1)
system = pd.read_csv(file_path2)
safety = pd.read_csv(file_path3)

# Identify common countries present in all three datasets
common_countries = set(prices['Country']) & set(system['Country']) & set(safety['Country'])

# Filter to common countries
prices_filtered = prices[prices['Country'].isin(common_countries)].copy()
system_filtered = system[system['Country'].isin(common_countries)].copy()
safety_filtered = safety[safety['Country'].isin(common_countries)].copy()

def map_safety_to_score(level):
    # level is 1 to 4
    mapping = {1:0, 2:3, 3:7, 4:10}
    return mapping.get(level, 5) # default if unexpected
safety_filtered['Safety_Score'] = safety_filtered['Risk'].apply(map_safety_to_score)

# Health system: (1 - WHOIndex) * 10
system_filtered['HealthSystem_Score'] = (1 - system_filtered['TheWorldWHOIndex2000']) * 10

# Normalize price to 0-10
min_price = prices_filtered['Healthcare prices'].min()
max_price = prices_filtered['Healthcare prices'].max()
def normalize_price(price):
    return ((price - min_price) / (max_price - min_price)) * 10
prices_filtered['Price_Score'] = prices_filtered['Healthcare prices'].apply(normalize_price)

# Create dictionaries for easy lookup
safety_dict = dict(zip(safety_filtered['Country'], safety_filtered['Safety_Score']))
system_dict = dict(zip(system_filtered['Country'], system_filtered['HealthSystem_Score']))
price_dict = dict(zip(prices_filtered['Country'], prices_filtered['Price_Score']))









# Analyze the top entries in the 'Destination' column
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load the city-to-country mapping
city_country_mapping = pd.read_csv(r'C:\Users\abadianov\Desktop\Projects\ML\city_to_country_mapping.csv')
city_country_dict = dict(zip(city_country_mapping["name"], city_country_mapping["Country"]))

# Load the airport-to-country mapping
airports = pd.read_csv(
    r'C:\Users\abadianov\Desktop\Projects\ML\airports_to_code_mapping.csv', 
    usecols=["IATA", "Country"]
)
airport_to_country_dict = dict(zip(airports["IATA"], airports["Country"]))

# Map 'Destination' to 'MappedDestination' using city and airport mappings
data['MappedDestination'] = (
    data['Destination']
    .map(city_country_dict)
    .fillna(data['Destination'].map(airport_to_country_dict))
    .fillna(data['Destination'])
)

# Initial standardization mapping
standardization_map = {
    "ca": "Canada", "CA": "Canada", "États-Unis": "United States", "US": "United States", "us": "United States",
    "United States of America": "United States", "United Kingdom": "UK", "GB": "UK", "gb": "UK", "France": "France",
    "fr": "France", "CN": "China", "cn": "China", "JP": "Japan", "jp": "Japan", "BR": "Brazil", "br": "Brazil",
    "Mexico": "Mexico", "mx": "Mexico", "Vietnam": "Vietnam", "vn": "Vietnam", "Québec": "Quebec",
    "QuÃ©bec": "Quebec", "Montréal": "Montreal", "MontrÃ©al": "Montreal", "Algérie": "Algeria",
    "Maroc": "Morocco", "Tunisie": "Tunisia", "Côte D'Ivoire": "Ivory Coast", "Inde": "India",
    "Chine": "China", "Japon": "Japan", "Corée du Sud": "South Korea", "Corée, République de": "South Korea",
    "Corée, République Populaire Démocratique de": "North Korea", "République Dominicaine": "Dominican Republic",
    "Indonésie": "Indonesia", "Sénégal": "Senegal", "Suisse": "Switzerland", "Émirats Arabes Unis": "United Arab Emirates",
    "Éthiopie": "Ethiopia", "Turquie": "Turkey", "Russie, Fédération de": "Russia"
}

data['MappedDestination'] = data['MappedDestination'].replace(standardization_map)
data['MappedDestination'] = data['MappedDestination'].str.title().fillna('Unknown')

data['SimplifiedDestination'] = data['MappedDestination']

# **Apply mappings after frequency thresholding**

# Combine manual mappings, region mappings, state/city mappings, and final replacements
combined_mappings = {
    # Manual mappings and abbreviations
    "Uk": "United Kingdom", "GB": "United Kingdom", "Gb": "United Kingdom", "Mx": "Mexico", "Es": "Spain",
    "Fr": "France", "It": "Italy", "Hk": "Hong Kong", "Au": "Australia", "In": "India", "Vn": "Vietnam",
    "Ph": "Philippines", "Do": "Dominican Republic", "Pt": "Portugal", "Cu": "Cuba", "Gr": "Greece",
    "Nl": "Netherlands", "Kr": "South Korea", "Kp": "North Korea", "Jm": "Jamaica", "Ie": "Ireland",
    "Tw": "Taiwan", "Th": "Thailand", "De": "Germany", "Br": "Brazil", "Cr": "Costa Rica", "Nz": "New Zealand",
    "Bb": "Barbados", "Ca": "Canada", "Us": "United States", "Il": "Israel", "Ch": "Switzerland",
    "Pl": "Poland", "Tr": "Turkey", "Ae": "United Arab Emirates", "Dk": "Denmark", "Ro": "Romania",
    "Se": "Sweden", "Hu": "Hungary", "Cl": "Chile", "Ar": "Argentina", "My": "Malaysia", "Id": "Indonesia",
    "Bs": "Bahamas", "Lb": "Lebanon", "Cw": "Curacao", "Rs": "Serbia", "Ag": "Antigua and Barbuda",
    "Lk": "Sri Lanka", "Ky": "Cayman Islands", "Ec": "Ecuador", "Sa": "Saudi Arabia", "Eg": "Egypt",
    "Cz": "Czech Republic", "Ke": "Kenya", "Lc": "Saint Lucia", "Bm": "Bermuda", "Co": "Colombia",
    "Lv": "Latvia", "Lt": "Lithuania", "Fi": "Finland", "Sk": "Slovakia", "Si": "Slovenia",
    "Bg": "Bulgaria", "Mt": "Malta", "By": "Belarus", "Ua": "Ukraine", "Md": "Moldova", "Am": "Armenia",
    "Az": "Azerbaijan", "Ge": "Georgia", "Kg": "Kyrgyzstan", "Kz": "Kazakhstan", "Uz": "Uzbekistan",
    "Tm": "Turkmenistan", "Tj": "Tajikistan", "Ir": "Iran", "Sy": "Syria", "Jo": "Jordan",
    "Pk": "Pakistan", "Bd": "Bangladesh", "Np": "Nepal", "Mg": "Madagascar", "Zm": "Zambia",
    "Zw": "Zimbabwe", "Mu": "Mauritius", "Mk": "North Macedonia", "Et": "Ethiopia", "Gh": "Ghana",
    "Sn": "Senegal", "Tn": "Tunisia", "Ng": "Nigeria", "Ma": "Morocco", "Za": "South Africa",
    "No": "Norway", "Is": "Iceland", "Be": "Belgium", "Hr": "Croatia", "Sg": "Singapore", "At": "Austria",
    "Pe": "Peru", "Pa": "Panama", "Fj": "Fiji", "Aw": "Aruba", "Tc": "Turks and Caicos Islands",
    "Yto": "Canada", "Ru": "Russia", "Bn": "Brunei", "Ua": "Ukraine",
    # French country names
    "Allemagne": "Germany", "Italie": "Italy", "Espagne": "Spain", "Suisse": "Switzerland",
    "Royaume-Uni": "United Kingdom", "Pays-Bas": "Netherlands", "Nouvelle-Zélande": "New Zealand",
    "Australie": "Australia", "États-Unis": "United States", "Émirats Arabes Unis": "United Arab Emirates",
    "Brésil": "Brazil", "Belgique": "Belgium", "Chine": "China", "Japon": "Japan", "Viêtnam": "Vietnam",
    "Corée Du Sud": "South Korea", "Corée, République De": "South Korea",
    "Russie, Fédération De": "Russia", "Mexique": "Mexico", "Irlande": "Ireland", "Grèce": "Greece",
    "Danemark": "Denmark", "Portugal": "Portugal", "Malaisie": "Malaysia", "Singapour": "Singapore",
    "Roumanie": "Romania", "Bulgarie": "Bulgaria", "Pologne": "Poland", "Autriche": "Austria",
    "Hongrie": "Hungary", "Norvège": "Norway", "Islande": "Iceland", "Finlande": "Finland",
    "Suède": "Sweden", "Maroc": "Morocco", "Afrique Du Sud": "South Africa", "Égypte": "Egypt",
    "Jamaïque": "Jamaica", "Iran, République Islamique D'": "Iran", "Arabie Saoudite": "Saudi Arabia",
    "Taïwan": "Taiwan", "Corée, République Populaire Démocratique De": "North Korea",
    "Dominique": "Dominica", "Congo/Kinshasa": "Democratic Republic of the Congo",
    "Congo": "Republic of the Congo", "Belize": "Belize", "Colombie": "Colombia",
    "Guinée": "Guinea", "Haïti": "Haiti", "Guadeloupe": "France", "Martinique": "France",
    "République Dominicaine": "Dominican Republic", "Trinité-Et-Tobago": "Trinidad and Tobago",
    "Cameroun": "Cameroon", "Gabon": "Gabon", "Mongolie": "Mongolia", "Mozambique": "Mozambique",
    "Maurice": "Mauritius", "Ouganda": "Uganda", "Zimbabwe": "Zimbabwe", "Zambie": "Zambia",
    "Bénin": "Benin", "Niger": "Niger", "Nigéria": "Nigeria", "Malawi": "Malawi", "Botswana": "Botswana",
    "Sénégal": "Senegal", "Seychelles": "Seychelles", "Rwanda": "Rwanda", "Swaziland": "Eswatini",
    "Namibie": "Namibia", "Ghana": "Ghana", "Éthiopie": "Ethiopia", "Équateur": "Ecuador",
    "Turquie": "Turkey", "Jordanie": "Jordan", "Israël": "Israel", "Liban": "Lebanon",
    "Syrienne, République Arabe": "Syria", "Koweït": "Kuwait", "Oman": "Oman", "Syrie": "Syria",
    "Emirats Arabes Unis": "United Arab Emirates", "Qatar": "Qatar", "Bahreïn": "Bahrain",
    "Argentine": "Argentina", "Chili": "Chile", "Pérou": "Peru", "Venezuela": "Venezuela",
    "Brunéi Darussalam": "Brunei", "Égypte": "Egypt", "Comores": "Comoros",
    # States and cities mapped to countries
    "Maine": "United States", "Missouri": "United States", "Kansas": "United States",
    "Connecticut": "United States", "Alabama": "United States", "Oklahoma": "United States",
    "Rhode Island": "United States", "West Virginia": "United States", "Mississippi": "United States",
    "California": "United States", "Arizona": "United States", "Georgia": "United States",
    "Nevada": "United States", "Hawaii": "United States", "Texas": "United States",
    "Michigan": "United States", "New York": "United States", "Ohio": "United States",
    "Tennessee": "United States", "Illinois": "United States", "Utah": "United States",
    "Indiana": "United States", "Idaho": "United States", "North Carolina": "United States",
    "Massachusetts": "United States", "Pennsylvania": "United States", "Minnesota": "United States",
    "Louisiana": "United States", "North Dakota": "United States", "South Carolina": "United States",
    "New Jersey": "United States", "Wisconsin": "United States", "Florida": "United States",
    "Montreal": "Canada", "Quebec": "Canada", "Quebec City": "Canada", "British Columbia": "Canada",
    "Alberta": "Canada", "Nova Scotia": "Canada", "Newfoundland & Labrador": "Canada",
    "Prince Edward Island": "Canada", "Saskatchewan": "Canada", "Manitoba": "Canada",
    "New Brunswick": "Canada", "Ontario": "Canada", "St Catharines": "Canada", "North York": "Canada",
    "Gatineau": "Canada", "Summerside": "Canada", "Yto": "Canada", "Laprairie": "Canada",
    "Cold Lake": "Canada", "Fort Langley": "Canada", "North Sydney": "Canada", "Kindersley": "Canada",
    "Lac La Biche": "Canada", "Priddis": "Canada", "St Adolphe": "Canada", "Cardston": "Canada",
    "Bonnyville": "Canada", "Strathmore": "Canada", "Dauphin": "Canada", "Watkins Glen": "United States",
    "Coraopolis": "United States", "Stateline": "United States", "Williamsville": "United States",
    # Other mappings
    "Puerto Rico": "United States", "Virgin Islands": "United States Virgin Islands",
    "Guadeloupe": "France", "Martinique": "France", "French Guiana": "France", "Reunion": "France",
    "Saint Martin": "France", "Saint Barthélemy": "France", "Saint-Martin": "France",
    "French Polynesia": "France", "Mayotte": "France", "Congo (Brazzaville)": "Republic of the Congo",
    "Congo (Kinshasa)": "Democratic Republic of the Congo",
    "Congo, La République Démocratique Du": "Democratic Republic of the Congo",
    "Congo": "Republic of the Congo", "Ivory Coast": "Côte d'Ivoire",
    "Ivory Coast (Côte D'Ivoire)": "Côte d'Ivoire", "Cote D'Ivoire": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde", "Greenland": "Denmark", "Timor-Leste": "East Timor",
    "Czechia": "Czech Republic", "Slovakia": "Slovakia", "Moldavie": "Moldova",
    "Kyrgyzstan": "Kyrgyzstan", "Uzbekistan": "Uzbekistan", "Tajikistan": "Tajikistan",
    "Turkmenistan": "Turkmenistan", "Kuwait": "Kuwait", "Bahrain": "Bahrain", "Qatar": "Qatar",
    "Ethiopie": "Ethiopia", "Algérie": "Algeria", "Soudan": "Sudan", "Kenya": "Kenya",
    "Mongolie": "Mongolia", "Bolivie": "Bolivia", "Ouganda": "Uganda", "Ukraine": "Ukraine",
    "Bélarus": "Belarus", "Belize": "Belize", "Comores": "Comoros",
    "Sierra Leone": "Sierra Leone", "Guinée": "Guinea", "Liban": "Lebanon", "Yémen": "Yemen",
    "Birmimgham": "United Kingdom",
    # Region mappings
    "World / Space / Other": "Other", "Unknown": "Other", "Enoch": "Other", "Ponoka": "Other",
    # Final replacements
    "Korea, Republic Of": "South Korea", "Iran, Islamic Republic Of": "Iran", "Inconnue": "Other",
    "Unspecified": "Other", "Great Britain": "United Kingdom",
}

data['SimplifiedDestination'] = data['SimplifiedDestination'].replace(combined_mappings)
data['SimplifiedDestination'] = data['SimplifiedDestination'].str.strip().str.title()
data['SimplifiedDestination'] = data['SimplifiedDestination'].apply(
    lambda x: 'Other' if any(char.isdigit() for char in x) else x
)

top_destinations = data['SimplifiedDestination'].value_counts().head(1000)
print(top_destinations)
print(f"Number of unique destinations: {data['SimplifiedDestination'].nunique()}")





# Define functions to get scores from dicts
def get_safety_score(country):
    return safety_dict.get(country, None)

def get_system_score(country):
    return system_dict.get(country, None)

def get_price_score(country):
    return price_dict.get(country, None)

# Map scores to the dataset
data['Safety_Score'] = data['SimplifiedDestination'].apply(get_safety_score)
data['System_Score'] = data['SimplifiedDestination'].apply(get_system_score)
data['Price_Score'] = data['SimplifiedDestination'].apply(get_price_score)

# Drop rows where any score is NaN
data = data.dropna(subset=['Safety_Score', 'System_Score', 'Price_Score'])

# Function to create risk categories with quantile-based bins
def create_risk_category_by_quantile(df, score_col):
    if df[score_col].nunique() <= 3:
        # Handle case with fewer unique values manually
        bins = [-float('inf'), 0.33 * df[score_col].max(), 0.66 * df[score_col].max(), float('inf')]
        labels = ['Low', 'Medium', 'High']
    else:
        q1 = df[score_col].quantile(0.33)
        q2 = df[score_col].quantile(0.66)
        bins = [df[score_col].min(), q1, q2, df[score_col].max()]
    
    # Ensure bins are unique
    bins = pd.unique(bins)
    labels = ['Low', 'Medium', 'High'][:len(bins) - 1]

    # Create categories with duplicate bins handled
    return pd.cut(df[score_col], bins=bins, labels=labels, include_lowest=True, duplicates='drop')

# Assign risk categories for scores
data['Safety_Risk_Category'] = create_risk_category_by_quantile(data, 'Safety_Score')
data['System_Risk_Category'] = create_risk_category_by_quantile(data, 'System_Score')
data['Price_Risk_Category'] = create_risk_category_by_quantile(data, 'Price_Score')

# Filter data to include only relevant rows
data = data[data['LossRatio'].notna()]  # Remove rows with missing LossRatio
data = data[data['ProductGroup'] == 'EHM']  # Filter by ProductGroup

# Quantify rows in each category
print("Safety Risk Category Counts:")
print(data['Safety_Risk_Category'].value_counts(dropna=False))

print("\nSystem Risk Category Counts:")
print(data['System_Risk_Category'].value_counts(dropna=False))

print("\nPrice Risk Category Counts:")
print(data['Price_Risk_Category'].value_counts(dropna=False))

# Generate pivot tables for LossRatio vs. Risk Categories
safety_pivot = (
    data.groupby('Safety_Risk_Category')['LossRatio']
    .mean()
    .reset_index()
)
print("\nSafety Risk Category vs LossRatio:")
print(safety_pivot)

system_pivot = (
    data.groupby('System_Risk_Category')['LossRatio']
    .mean()
    .reset_index()
)
print("\nSystem Risk Category vs LossRatio:")
print(system_pivot)

price_pivot = (
    data.groupby('Price_Risk_Category')['LossRatio']
    .mean()
    .reset_index()
)
print("\nPrice Risk Category vs LossRatio:")
print(price_pivot)





# Define the top destinations to keep as separate categories
top_countries = [
    'Canada', 'United States', 'Mexico', 'Cuba'
]

# Combine top countries and high-risk countries to exclude from regional mapping
excluded_countries = set(top_countries)

# Create a mapping of countries to regions
country_to_region = {
    # Europe
    'United Kingdom': 'Europe', 'France': 'Europe', 'Germany': 'Europe', 'Spain': 'Europe',
    'Italy': 'Europe', 'Portugal': 'Europe', 'Greece': 'Europe', 'Netherlands': 'Europe',
    'Ireland': 'Europe', 'Poland': 'Europe', 'Switzerland': 'Europe', 'Austria': 'Europe',
    'Denmark': 'Europe', 'Hungary': 'Europe', 'Sweden': 'Europe', 'Belgium': 'Europe',
    'Norway': 'Europe', 'Romania': 'Europe', 'Czech Republic': 'Europe', 'Serbia': 'Europe',
    'Finland': 'Europe', 'Iceland': 'Europe', 'Malta': 'Europe', 'Croatia': 'Europe',
    'Slovakia': 'Europe', 'Slovenia': 'Europe', 'Lithuania': 'Europe', 'Latvia': 'Europe',
    'Estonia': 'Europe', 'Ukraine': 'Europe', 'Belarus': 'Europe', 'Moldova': 'Europe',
    'Bosnia And Herzegovina': 'Europe', 'North Macedonia': 'Europe', 'Albania': 'Europe',
    'Montenegro': 'Europe', 'Russia': 'Europe',

    # Latin America and the Caribbean
    'Brazil': 'Latin America and Caribbean', 'Costa Rica': 'Latin America and Caribbean',
    'Jamaica': 'Latin America and Caribbean', 'Colombia': 'Latin America and Caribbean',
    'Barbados': 'Latin America and Caribbean', 'Bahamas': 'Latin America and Caribbean',
    'Panama': 'Latin America and Caribbean', 'Peru': 'Latin America and Caribbean',
    'Chile': 'Latin America and Caribbean', 'Venezuela': 'Latin America and Caribbean',
    'Guatemala': 'Latin America and Caribbean', 'Saint Lucia': 'Latin America and Caribbean',
    'Turks And Caicos Islands': 'Latin America and Caribbean', 'Cayman Islands': 'Latin America and Caribbean',
    'Antigua And Barbuda': 'Latin America and Caribbean', 'Honduras': 'Latin America and Caribbean',
    'Bermuda': 'Latin America and Caribbean', 'Belize': 'Latin America and Caribbean',
    'El Salvador': 'Latin America and Caribbean', 'Grenada': 'Latin America and Caribbean',
    'Guyana': 'Latin America and Caribbean', 'Saint Martin': 'Latin America and Caribbean',
    'Trinidad And Tobago': 'Latin America and Caribbean', 'Dominica': 'Latin America and Caribbean',
    'Haiti': 'Latin America and Caribbean', 'Paraguay': 'Latin America and Caribbean',
    'Bolivia': 'Latin America and Caribbean', 'Curacao': 'Latin America and Caribbean',
    'Uruguay': 'Latin America and Caribbean', 'Nicaragua': 'Latin America and Caribbean',
    'Dominican Republic': 'Latin America and Caribbean', 'Argentina': 'Latin America and Caribbean',
    'Suriname': 'Latin America and Caribbean', 'French Guiana': 'Latin America and Caribbean',
    'Saint Kitts And Nevis': 'Latin America and Caribbean', 'Saint Vincent And The Grenadines': 'Latin America and Caribbean',
    'Montserrat': 'Latin America and Caribbean', 'Anguilla': 'Latin America and Caribbean',
    'Aruba': 'Latin America and Caribbean', 'British Virgin Islands': 'Latin America and Caribbean',
    'Guadeloupe': 'Latin America and Caribbean', 'Martinique': 'Latin America and Caribbean',
    'Saint Barthélemy': 'Latin America and Caribbean',

    # Asia
    'China': 'Asia', 'Japan': 'Asia', 'India': 'Asia', 'South Korea': 'Asia', 'Philippines': 'Asia',
    'Vietnam': 'Asia', 'Hong Kong': 'Asia', 'Thailand': 'Asia', 'Taiwan': 'Asia',
    'Indonesia': 'Asia', 'Malaysia': 'Asia', 'Singapore': 'Asia', 'Sri Lanka': 'Asia',
    'Cambodia': 'Asia', 'Pakistan': 'Asia', 'Bangladesh': 'Asia', 'North Korea': 'Asia',
    'Nepal': 'Asia', 'Kazakhstan': 'Asia', 'Kyrgyzstan': 'Asia', 'Uzbekistan': 'Asia',
    'Turkmenistan': 'Asia', 'Tajikistan': 'Asia', 'Mongolia': 'Asia', 'Myanmar': 'Asia',
    'Laos': 'Asia', 'Bhutan': 'Asia', 'Maldives': 'Asia', 'Brunei': 'Asia', 'Afghanistan': 'Asia',

    # Middle East
    'United Arab Emirates': 'Middle East', 'Saudi Arabia': 'Middle East', 'Qatar': 'Middle East',
    'Kuwait': 'Middle East', 'Bahrain': 'Middle East', 'Oman': 'Middle East', 'Israel': 'Middle East',
    'Jordan': 'Middle East', 'Lebanon': 'Middle East', 'Turkey': 'Middle East', 'Iran': 'Middle East',
    'Iraq': 'Middle East', 'Syria': 'Middle East', 'Yemen': 'Middle East', 'Palestine': 'Middle East',

    # Africa
    'South Africa': 'Africa', 'Egypt': 'Africa', 'Morocco': 'Africa', 'Kenya': 'Africa',
    'Tunisia': 'Africa', 'Nigeria': 'Africa', 'Algeria': 'Africa', 'Ethiopia': 'Africa',
    'Ghana': 'Africa', 'Uganda': 'Africa', 'Tanzania': 'Africa', 'Zimbabwe': 'Africa',
    'Zambia': 'Africa', 'Senegal': 'Africa', 'Ivory Coast': 'Africa', "Côte D'Ivoire": 'Africa',
    'Cameroon': 'Africa', 'Botswana': 'Africa', 'Namibia': 'Africa', 'Mozambique': 'Africa',
    'Madagascar': 'Africa', 'Rwanda': 'Africa', 'Sudan': 'Africa', 'Democratic Republic Of The Congo': 'Africa',
    'Republic Of The Congo': 'Africa', 'Angola': 'Africa', 'Mali': 'Africa', 'Mauritius': 'Africa',
    'Seychelles': 'Africa', 'Gabon': 'Africa', 'Benin': 'Africa', 'Burkina Faso': 'Africa',
    'Malawi': 'Africa', 'Liberia': 'Africa', 'Sierra Leone': 'Africa', 'Niger': 'Africa',
    'Guinea': 'Africa', 'Equatorial Guinea': 'Africa', 'Togo': 'Africa', 'Eritrea': 'Africa',
    'Chad': 'Africa', 'Somalia': 'Africa', 'Central African Republic': 'Africa', 'Gambia': 'Africa',
    'Burundi': 'Africa', 'Lesotho': 'Africa', 'Djibouti': 'Africa', 'Eswatini': 'Africa',
    'South Sudan': 'Africa', 'Cabo Verde': 'Africa', 'Comoros': 'Africa', 'Sao Tome And Principe': 'Africa',

    # Oceania
    'Australia': 'Oceania', 'New Zealand': 'Oceania', 'Fiji': 'Oceania', 'Papua New Guinea': 'Oceania',
    'Samoa': 'Oceania', 'Tonga': 'Oceania', 'Vanuatu': 'Oceania', 'Solomon Islands': 'Oceania',
    'Micronesia': 'Oceania', 'Kiribati': 'Oceania', 'Tuvalu': 'Oceania', 'Palau': 'Oceania',
    'Marshall Islands': 'Oceania', 'Nauru': 'Oceania',

    # Other
    'Antarctica': 'Other', 'Greenland': 'Other', 'Unknown': 'Other', 'Other': 'Other',
}

# Function to map countries to regions
def map_country_to_region(country):
    if country in excluded_countries:
        return country  # Keep as is
    else:
        return country_to_region.get(country, 'Other')

# Apply the mapping to create 'DestinationRegion' column
data['DestinationRegion'] = data['SimplifiedDestination'].apply(map_country_to_region)

# Normalize case and strip whitespace (if necessary)
data['DestinationRegion'] = data['DestinationRegion'].str.strip().str.title()

# Verify the counts for each region and country
# Filter data for 'EHM' ProductGroup
region_counts = data['DestinationRegion'].value_counts()
print(region_counts)
print(f"Number of unique regions/destinations: {data['DestinationRegion'].nunique()}")


# Saving the updated CSV file
# Extract folder path and file name
folder_path, original_file_name = os.path.split(file_path)
file_name, file_extension = os.path.splitext(original_file_name)

# Create a new file name by appending "_updated_dest" before the extension
new_file_name = f"{file_name}_updated_dest{file_extension}"
new_file_path = os.path.join(folder_path, new_file_name)

# Save the DataFrame to the new file
data.to_csv(new_file_path, index=False)

print(f"Updated file saved to: {new_file_path}")
