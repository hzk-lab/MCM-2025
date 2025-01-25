import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load the datasets
results_df = pd.read_csv('updated_silver_medals.csv', encoding='ISO-8859-1')
programs_df = pd.read_csv('summerOly_programs.csv', encoding='ISO-8859-1')

athletics_results = results_df[results_df['Sport'] == 'Table Tennis'] #Change sport name here

# Ensure the 'summerOly_programs.csv' DataFrame has the necessary column names
# Rename columns in programs_df to match year format if necessary
programs_df.rename(columns=lambda x: str(x).strip(), inplace=True)

# Create a list to store country's gold medal percentage for each year
country_gold_percentage_medals = []

# Loop through each year and calculate the percentage of gold medals won by country
for year in athletics_results['Year'].unique():
    year_data = athletics_results[athletics_results['Year'] == year]
    
    # Locate the total number of gold medals for athletics in this year from programs_df
    sport_program_data = programs_df[programs_df['Discipline'] == 'Table Tennis']  # Adjust 'Discipline' if necessary

    if not sport_program_data.empty and str(year) in sport_program_data.columns:
    # Convert the value to numeric to ensure it's treated as a number
        try:
            total_gold_medals_in_year = pd.to_numeric(sport_program_data[str(year)].sum(), errors='coerce')
        except ValueError:
            total_gold_medals_in_year = 0  # Default to 0 if conversion fails
    else:
        total_gold_medals_in_year = 0   
    
    
    # Get the number of gold medals won by country
    country_gold_medals = year_data['CHN'].sum()  # 'CHN' column represents country    "Change country name here"
    print(f"Year: {year}, Total Gold Medals: {total_gold_medals_in_year}, country's Gold Medals: {country_gold_medals}")
    
    
    # Calculate the percentage of total gold medals won by country
    if total_gold_medals_in_year > 0:  # Prevent division by zero
        country_gold_percentage = (country_gold_medals / total_gold_medals_in_year) * 100
    else:
        country_gold_percentage = 0  # If no gold medals were awarded in that year

    # Store the result with the year

    country_gold_percentage_medals.append({
        'Year': year,
        'Country_Gold_Percentage': country_gold_percentage
    })

# Create a DataFrame with the results
country_gold_percentage_df = pd.DataFrame(country_gold_percentage_medals)

# Plotting the trend of country's gold medal percentage over the years
plt.figure(figsize=(10, 6))
plt.scatter(country_gold_percentage_df['Year'], country_gold_percentage_df['Country_Gold_Percentage'], color='gold', label='Actual Data')

# Adding labels and title
plt.title("Country's Gold Medal Percentage", fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Gold Medal Percentage', fontsize=12)
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
