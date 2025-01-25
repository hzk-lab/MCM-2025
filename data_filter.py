import pandas as pd

# Load the dataset
results_df = pd.read_csv('updated_bronze_medals.csv', encoding='ISO-8859-1')

# Check if 'URS' and 'ROC' are in the columns
countries = results_df.columns[3:]  # Assuming the first 3 columns are non-country columns
if 'URS' in countries and 'ROC' in countries:
    # Combine 'URS' and 'ROC' medals
    results_df['ROC'] = results_df['URS'] + results_df['ROC']
    
    # Drop the 'URS' column since we no longer need it
    results_df.drop(columns=['URS'], inplace=True)

    # Save the updated dataset to a new file
    results_df.to_csv('updated_bronze_medals.csv', index=False)
    print("Updated medal distribution saved to 'updated_gold_medals_combined_roc.csv'.")
else:
    print("'URS' or 'ROC' not found in the dataset.")
