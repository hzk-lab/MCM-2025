import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# 1. Load data
file_path = '/Users/zekaihuang/Desktop/MCM2025/updated_gold_medals_int.csv'
data = pd.read_csv(file_path)

# Inspect the first few rows to confirm structure
print(data.head())

# 2. Data preprocessing
# Extract country codes (skip 'Year' and 'Sport')
country_codes = data.columns[2:]  # Assuming the first two columns are 'Year' and 'Sport'

# Convert the data from wide format to long format
long_data = data.melt(
    id_vars=["Year", "Sport"],
    value_vars=country_codes,
    var_name="Country",
    value_name="Gold_Medals",
)

# Fill missing values with 0
long_data["Gold_Medals"].fillna(0, inplace=True)

# Convert 'Year' to integer (if it's not already)
long_data["Year"] = long_data["Year"].astype(int)

# Get unique countries and years
countries = long_data["Country"].unique()
years = long_data["Year"].unique()

# 3. Build hierarchical Bayesian model
with pm.Model() as hierarchical_model:
    # Global prior (overall trend)
    global_mean = pm.Normal("global_mean", mu=10, sigma=5)
    sigma = pm.Exponential("sigma", lam=0.1)
    
    # Country-level parameters
    country_offset = pm.Normal("country_offset", mu=0, sigma=1, shape=len(countries))
    year_offset = pm.Normal("year_offset", mu=0, sigma=1, shape=len(years))

    # Indices for country and year
    country_idx = long_data["Country"].apply(lambda x: np.where(countries == x)[0][0]).values
    year_idx = long_data["Year"].apply(lambda x: np.where(years == x)[0][0]).values

    # Medal count modeled with Poisson distribution
    mu = pm.math.exp(global_mean + country_offset[country_idx] + year_offset[year_idx])
    medals = pm.Poisson("medals", mu=mu, observed=long_data["Gold_Medals"].values)

    # Sampling with increased target_accept
    trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)


# 5. Predict for 2028
# Extend the year index to include 2028
new_year_idx = np.where(years == 2028)[0][0] if 2028 in years else len(years)
years = np.append(years, 2028)  # Update the years array

# Calculate the posterior predictions for 2028
country_offsets = trace.posterior["country_offset"].values  # Posterior samples for country offset
year_offsets = trace.posterior["year_offset"].values  # Posterior samples for year offset
global_mean_samples = trace.posterior["global_mean"].values  # Global mean samples

# Initialize predictions
predictions_2028 = []

# Loop through each country to predict for 2028
for i, country in enumerate(countries):
    # Get the posterior predictions for each country
    pred_2028 = global_mean_samples + country_offsets[:, :, i] + year_offsets[:, :, new_year_idx - 1]
    predictions_2028.append(pred_2028.mean(axis=(0, 1)))  # Mean prediction

# Create a DataFrame for predictions
predictions_df = pd.DataFrame({
    "Country": countries,
    "Predicted_Gold_Medals_2028": predictions_2028
})

# Sort predictions by the number of gold medals
predictions_df = predictions_df.sort_values(by="Predicted_Gold_Medals_2028", ascending=False)

output_file = '/Users/zekaihuang/Desktop/MCM2025/predicted_gold_medals_2028.csv'
predictions_df.to_csv(output_file, index=False)

