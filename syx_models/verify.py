# Prepare data for modeling
medal_distribution = []

# Iterate through each year
for year in results_df['Year'].unique():
    year_data = results_df[results_df['Year'] == year]  # Filter data for the specific year
    total_medals = year_data.iloc[:, 3:].sum().sum()  # Total medals across all countries
    
    # Sum medals for each country
    country_medals = year_data.iloc[:, 3:].sum()  # Sum medals for each country
    country_medals['Year'] = year  # Add year as a column to the country medals
    medal_distribution.append(country_medals)

# Convert the medal distribution list to a DataFrame
total_medal_counts_df = pd.DataFrame(medal_distribution)

# Sum the total medals for each country across all years
total_medals_per_country = total_medal_counts_df.sum(axis=0).sort_values(ascending=False)

# Convert to a DataFrame for better visualization
total_medals_per_country_df = pd.DataFrame(total_medals_per_country, columns=['Total_Medals'])

# Save the results to a CSV file
total_medals_per_country_df.to_csv('total_medals_per_country.csv', index=True)
print(total_medals_per_country_df)
