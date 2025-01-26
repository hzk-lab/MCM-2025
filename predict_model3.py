import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load datasets
results_df = pd.read_csv('updated_gold_medals.csv', encoding='ISO-8859-1')
programs_df = pd.read_csv('summerOly_programs.csv', encoding='ISO-8859-1')

# Prepare data
programs_df.rename(columns=lambda x: str(x).strip(), inplace=True)
sports = results_df['Sport'].unique()
programs_df = programs_df[programs_df['2024'].notna()]  # Ensure events exist for 2024

# Initialize a medal table
medal_table = {}

# Process each sport
for sport in sports:
    sport_results = results_df[results_df['Sport'] == sport]
    sport_programs = programs_df[programs_df['Sport'] == sport]
    countries = sport_results.columns[3:]

    # Prepare medal distribution data for this sport
    medal_distribution = []
    for year in sport_results['Year'].unique():
        year_data = sport_results[sport_results['Year'] == year]
        total_medals = year_data[countries].sum().sum()

        # Calculate percentage of medals for each country
        percentages = (year_data[countries].sum() / total_medals * 100).to_dict() if total_medals > 0 else {country: 0 for country in countries}
        percentages['Year'] = year
        medal_distribution.append(percentages)

    # Convert to DataFrame
    medal_df = pd.DataFrame(medal_distribution)

    # Reshape the data for LSTM model (needs 3D input: [samples, time_steps, features])
    X = medal_df['Year'].values.reshape(-1, 1, 1)  # reshape to (n_samples, n_time_steps, n_features)
    predictions = {}

    # MinMaxScaler for scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Predict for each country using LSTM
    for country in countries:
        y = medal_df[country].fillna(0).values.reshape(-1, 1)
        y_scaled = scaler.fit_transform(y)  # Scale the labels to range [0,1]

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dense(1))  # Output layer
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X, y_scaled, epochs=200, batch_size=1, verbose=0)

        # Predict the value for the next year (2024)
        next_year = np.array([[2024]]).reshape(1, 1, 1)  # reshape for LSTM input
        prediction = model.predict(next_year)
        
        # Inverse transform to get back the actual scale
        predictions[country] = scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]

    # Normalize predictions to sum to 100%
    total_prediction = sum(predictions.values())
    predictions = {country: (predictions[country] / total_prediction) * 100 for country in predictions}

    # Get total gold medals for this sport
    total_gold_medals_2024 = sport_programs['2024'].sum()

    # Calculate predicted medal counts
    predicted_medals = {country: (percentage / 100) * total_gold_medals_2024 for country, percentage in predictions.items()}
    predicted_medals_df = pd.DataFrame({
        'Country': predicted_medals.keys(),
        'Predicted_Medals': predicted_medals.values()
    })

    # Round down and handle residuals
    predicted_medals_df['Rounded_Medals'] = predicted_medals_df['Predicted_Medals'].apply(np.floor).astype(int)
    predicted_medals_df['Residual'] = predicted_medals_df['Predicted_Medals'] - predicted_medals_df['Rounded_Medals']
    shortfall = int(total_gold_medals_2024 - predicted_medals_df['Rounded_Medals'].sum())

    if shortfall > 0:
        top_residual_indices = predicted_medals_df.nlargest(shortfall, 'Residual').index
        predicted_medals_df.loc[top_residual_indices, 'Rounded_Medals'] += 1

    # Update medal table
    for _, row in predicted_medals_df.iterrows():
        country = row['Country']
        medals = row['Rounded_Medals']

        # Exclude ROC except for specific sports
        if country == 'Russia' and sport not in ['Athletics', 'Swimming', 'Gymnastics']:
            medals = 0

        medal_table[country] = medal_table.get(country, 0) + medals

# Convert medal table to DataFrame and sort
final_medal_table = pd.DataFrame({
    'Country': medal_table.keys(),
    'Total_Gold_Medals': medal_table.values()
}).sort_values(by='Total_Gold_Medals', ascending=False)

# Save the results
final_medal_table.to_csv('predicted_medal_distribution_2024_lstm.csv', index=False)
print(final_medal_table)
