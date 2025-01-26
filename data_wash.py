import pandas as pd

# Load the dataset
file_path = '/Users/zekaihuang/Desktop/MCM2025/updated_gold_medals.csv'  # Replace with the actual path to your file
data = pd.read_csv(file_path)

# Identify numeric columns
numeric_columns = data.select_dtypes(include=['float', 'int']).columns

# Convert only numeric columns to integers
data[numeric_columns] = data[numeric_columns].astype(int)

# Save the modified DataFrame
output_path = '/Users/zekaihuang/Desktop/MCM2025/updated_gold_medals_int.csv'  # Replace with the desired output path
data.to_csv(output_path, index=False)

print(f"Modified file saved at: {output_path}")
