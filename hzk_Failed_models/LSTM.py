import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# 1. Load the data
df = pd.read_csv('C:/Users/leonhuangzekai/Desktop/MCM/MCM-2025/updated_gold_medals_int.csv')

# 2. Transform from wide to long format:
#    - If your data has columns: ['Year', 'Sport', 'AIN', 'ALG', 'ANZ', ...]
#    - You can do something like:
id_vars = ['Year', 'Sport']
value_vars = [col for col in df.columns if col not in id_vars]  # all the country columns
long_df = pd.melt(
    df,
    id_vars=id_vars,
    value_vars=value_vars,
    var_name='Country',
    value_name='GoldMedals'
)

# Assume long_df is your main dataframe: [Year, Sport, Country, GoldMedals, (other features)...]

# Step A: Sort by (Country, Sport, Year)
long_df = long_df.sort_values(by=['Country', 'Sport', 'Year'])

# Step B: For each (Country, Sport), build sequences
grouped = []
time_steps = 3  # Example, you might want to use more Olympics in each sequence

for (country, sport), subdf in long_df.groupby(['Country', 'Sport']):
    subdf = subdf.reset_index(drop=True)
    
    # We must ensure subdf is in ascending chronological order
    # subdf has columns: Year, GoldMedals, plus any other features you want
    for i in range(len(subdf) - time_steps):
        # X_seq: from i to i+time_steps-1
        X_seq = subdf.iloc[i : i+time_steps]
        
        # Y: the next year's gold medals (position i+time_steps)
        y_val = subdf['GoldMedals'].iloc[i+time_steps]
        
        # Gather the features from X_seq as a numpy array
        # e.g., use: ['GoldMedals', 'Feature1', 'Feature2', ...]
        # But exclude 'Year' and your label columns from the input
        feature_cols = [...]  # define your feature list
        X_features = X_seq[feature_cols].values
        
        grouped.append((X_features, y_val))

# Step C: Convert to final numpy arrays
X_list = []
y_list = []
for (X_seq, y_val) in grouped:
    X_list.append(X_seq)
    y_list.append(y_val)

X_all = np.array(X_list)  # shape = (num_samples, time_steps, num_features)
y_all = np.array(y_list)  # shape = (num_samples,)

# Step D: Train-Test Split (time-based or random, depending on your approach)
split_idx = int(len(X_all) * 0.8)
X_train, y_train = X_all[:split_idx], y_all[:split_idx]
X_test,  y_test  = X_all[split_idx:], y_all[split_idx:]

# Step E: Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(time_steps, len(feature_cols))))
model.add(Dense(1, activation='linear'))  # for regression of gold medals

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

# Step F: Train
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=50,
          batch_size=32)

# Step G: Predict
y_pred_lstm = model.predict(X_test)
