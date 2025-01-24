import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('summerOly_medal_counts.csv')

# Clean column names (optional, in case there are extra spaces)
data.columns = data.columns.str.strip()

# Filter for China's data
china_data = data[data['NOC'] == 'United States']     #put in country name here.

# Check if China data exists
if not china_data.empty:
    # Extract year and medal counts
    china_years = china_data['Year']
    china_gold = china_data['Gold']
    china_silver = china_data['Silver']
    china_bronze = china_data['Bronze']

    # Create the dot plot
    plt.figure(figsize=(10, 6))

    # Plot gold, silver, and bronze
    plt.scatter(china_years, china_gold, color='gold', label='Gold', s=50)
    plt.scatter(china_years, china_silver, color='silver', label='Silver', s=50)
    plt.scatter(china_years, china_bronze, color='brown', label='Bronze', s=50)

    # Add labels and legend
    plt.title('United States\'s Medal Counts Across Olympic Years', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Medal Counts', fontsize=12)
    plt.xticks(china_years, rotation=45)
    plt.legend(title='Medal Type')
    plt.grid(alpha=0.5, linestyle='--')

    # Show the plot
    plt.tight_layout()
    plt.show()
else:
    print("No data found for China (NOC: 'CHN').")
