import pandas as pd

# Load the dataset
file_path = '/Users/zekaihuang/Desktop/MCM/MCM2025/Documents/Y2sem2/2025_Problem_C_Data/gold_medals_by_sport_and_country.csv'  # Replace with the actual path to your file
data = pd.read_csv(file_path)

# Identify numeric columns
numeric_columns = data.select_dtypes(include=['float', 'int']).columns

# Convert only numeric columns to integers
data[numeric_columns] = data[numeric_columns].astype(int)

# Save the modified DataFrame
output_path = '/Users/zekaihuang/Desktop/MCM/MCM2025/Documents/Y2sem2/2025_Problem_C_Data/gold_medals_by_sport_and_country_updated.csv'  # Replace with the desired output path
data.to_csv(output_path, index=False)

print(f"Modified file saved at: {output_path}")
