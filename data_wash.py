import pandas as pd

# Load the dataset
file_path = 'C:/Users/leonhuangzekai/Desktop/MCM/updated_total_medals.csv'  # Replace with the actual path to your file
data = pd.read_csv(file_path)

# Identify numeric columns
numeric_columns = data.select_dtypes(include=['float', 'int']).columns

# Handle non-finite values (NaN or inf) by filling or replacing them
data[numeric_columns] = data[numeric_columns].fillna(0)  # Replace NaN with 0
data[numeric_columns] = data[numeric_columns].replace([float('inf'), -float('inf')], 0)  # Replace inf/-inf with 0

# Convert only numeric columns to integers
data[numeric_columns] = data[numeric_columns].astype(int)

# Save the modified DataFrame
output_path = 'C:/Users/leonhuangzekai/Desktop/MCM/updated_total_medals_int.csv'  # Replace with the desired output path
data.to_csv(output_path, index=False)

print(f"Modified file saved at: {output_path}")
