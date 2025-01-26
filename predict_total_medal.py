import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the datasets
results_df = pd.read_csv('updated_gold_medals.csv', encoding='ISO-8859-1')
programs_df = pd.read_csv('summerOly_programs.csv', encoding='ISO-8859-1')

# Preprocess column names
programs_df.rename(columns=lambda x: str(x).strip(), inplace=True)
results_df.rename(columns=lambda x: str(x).strip(), inplace=True)

# Get all sports
sports = results_df['Sport'].unique()

# Initialize a dictionary to store total medals per country
total_medals_per_country = {}

# Process each sport
for sport in sports:
    print(f"Processing {sport}...")

    # Filter the data for the current sport
    sport_results = results_df[results_df['Sport'] == sport]
    countries = sport_results.columns[3:]  # Assuming country medal counts start at column index 3

    # Prepare medal distribution data
    medal_distribution = []
    for year in sport_results['Year'].unique():
        year_data = sport_results[sport_results['Year'] == year]
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
        degree = 12  # Adjust degree as needed
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

    # Get total gold medals for the sport in 2024
    sport_events = programs_df[programs_df['Sport'] == sport]
    total_gold_medals_2024 = sport_events['2024'].sum()  # Sum across all disciplines within the sport

    # Calculate the predicted number of medals for each country
    predicted_medals = {country: (percentage / 100) * total_gold_medals_2024 for country, percentage in predictions.items()}

    # Convert to DataFrame for rounding
    predicted_medals_df = pd.DataFrame({
        'Country': predicted_medals.keys(),
        'Predicted_Medals': predicted_medals.values()
    })
    predicted_medals_df['Rounded_Medals'] = predicted_medals_df['Predicted_Medals'].apply(np.floor).astype(int)
    predicted_medals_df['Residual'] = predicted_medals_df['Predicted_Medals'] - predicted_medals_df['Rounded_Medals']

    # Handle shortfall
    shortfall = int(total_gold_medals_2024 - predicted_medals_df['Rounded_Medals'].sum())
    if shortfall > 0:
        top_residual_indices = predicted_medals_df.nlargest(shortfall, 'Residual').index
        predicted_medals_df.loc[top_residual_indices, 'Rounded_Medals'] += 1

    # Update total medals per country
    for _, row in predicted_medals_df.iterrows():
        country = row['Country']
        rounded_medals = row['Rounded_Medals']
        total_medals_per_country[country] = total_medals_per_country.get(country, 0) + rounded_medals

# Convert total medals to a DataFrame
total_medals_df = pd.DataFrame(list(total_medals_per_country.items()), columns=['Country', 'Total_Medals'])
total_medals_df = total_medals_df.sort_values(by='Total_Medals', ascending=False)

# Save to CSV
total_medals_df.to_csv('total_predicted_gold_medals_2024.csv', index=False)
print("Final total predicted gold medals for 2024 saved to 'total_predicted_gold_medals_2024.csv'.")

# Visualize the total predicted medal counts
plt.figure(figsize=(14, 8))
plt.bar(total_medals_df['Country'], total_medals_df['Total_Medals'], color='gold')
plt.title("Predicted Total Gold Medals for All Sports in 2024", fontsize=16)
plt.xlabel("Country", fontsize=12)
plt.ylabel("Total Gold Medals", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
