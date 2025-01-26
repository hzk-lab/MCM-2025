import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the datasets
results_df = pd.read_csv('updated_gold_medals.csv', encoding='ISO-8859-1')
programs_df = pd.read_csv('summerOly_programs.csv', encoding='ISO-8859-1')

# Filter the data for the selected sport
selected_sport = 'Swimming'  # Change sport name here
athletics_results = results_df[results_df['Sport'] == selected_sport]
programs_df.rename(columns=lambda x: str(x).strip(), inplace=True)

# Get all unique countries
countries = athletics_results.columns[3:]  # Assuming country medal counts start at column index 3

# Prepare data for modeling
medal_distribution = []

for year in athletics_results['Year'].unique():
    year_data = athletics_results[athletics_results['Year'] == year]
    total_medals = year_data[countries].sum().sum()  # Total medals in that year

    # Calculate percentage of medals for each country
    if total_medals > 0:
        percentages = (year_data[countries].sum() / total_medals * 100).to_dict()
    else:
        percentages = {country: 0 for country in countries}

    percentages['Year'] = year
    medal_distribution.append(percentages)

# Convert to a DataFrame
medal_df = pd.DataFrame(medal_distribution)

# Prediction for each country
X = medal_df['Year'].values.reshape(-1, 1)
predictions = {}

for country in countries:
    y = medal_df[country].fillna(0).values  # Fill missing values with 0

    # Polynomial regression
    degree = 2  # Adjust degree as needed
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict for the next year (2024)
    next_year = 2024
    next_year_poly = poly_features.transform([[next_year]])
    predictions[country] = max(0, model.predict(next_year_poly)[0])  # Ensure no negative predictions

# Normalize predictions to sum to 100%
total_prediction = sum(predictions.values())
for country in predictions:
    predictions[country] = (predictions[country] / total_prediction) * 100

# Convert predictions to a DataFrame for visualization
predicted_df = pd.DataFrame({
    'Country': predictions.keys(),
    'Predicted_Percentage': predictions.values()
}).sort_values(by='Predicted_Percentage', ascending=False)

# Save predictions to CSV
predicted_df.to_csv('predicted_medal_distribution_2024.csv', index=False)
print("Predicted medal distribution for 2024 saved to 'predicted_medal_distribution_2024.csv'.")

# Visualize the predicted distribution
plt.figure(figsize=(12, 6))
plt.bar(predicted_df['Country'], predicted_df['Predicted_Percentage'], color='skyblue')
plt.title(f"Predicted Medal Distribution for {selected_sport} in {next_year}", fontsize=16)
plt.xlabel("Country", fontsize=12)
plt.ylabel("Percentage of Medals", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



