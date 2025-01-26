import pandas as pd

# Load the dataset
athletes_data = pd.read_csv("summerOly_athletes.csv")

# Filter only gold medal winners
gold_medals = athletes_data[athletes_data['Medal'] == 'Bronze']

# Group by Year, Sport, Event (e.g., 'Men's Singles', 'Men's Doubles'), and Country to count unique gold medals per event
gold_counts = gold_medals.drop_duplicates(subset=['Year', 'Sport', 'Event', 'NOC']).groupby(['Year', 'Sport', 'Event', 'NOC']).size().reset_index(name='Bronze_Medals')

# Group by Year and Sport, summing the gold medals for each country in the respective sport
gold_counts_sport = gold_counts.groupby(['Year', 'Sport', 'NOC'])['Bronze_Medals'].sum().reset_index()

# Pivot the table to have countries as columns
pivot_table = gold_counts_sport.pivot_table(index=['Year', 'Sport'], 
                                            columns='NOC', 
                                            values='Bronze_Medals', 
                                            fill_value=0).reset_index()

# Save the updated table to a CSV file
pivot_table.to_csv("updated_bronze_medals.csv", index=False)

print("File 'updated_bronze_medals.csv' created successfully!")
