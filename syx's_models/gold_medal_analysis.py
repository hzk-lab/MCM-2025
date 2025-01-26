import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Load the data
data = pd.read_csv("summerOly_medal_counts.csv")

# Extract x and y columns
china_data = data[data['NOC'] == 'China']

x = china_data['Year']
y = china_data['Gold']

# Define the linear model
def linear_model(x, m, b):
    return m * x + b

# Fit the model
params, _ = curve_fit(linear_model, x, y)
m, b = params
print(f"Model: y = {m:.2f}x + {b:.2f}")

# Evaluate the model
y_pred = linear_model(x, m, b)
r2 = r2_score(y, y_pred)
print(f"R-squared: {r2:.2f}")

# Plot the data and fitted model
plt.scatter(x, y, label="Data")
plt.plot(x, y_pred, color='red', label="Fitted Model")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
