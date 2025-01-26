#!/usr/bin/env python3
"""
Example pipeline to predict future Olympic gold medals per (Country, Sport, Year)
using a tree-based model (XGBoost). Adapt as needed.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import math

class SafeLabelEncoder(LabelEncoder):
    def __init__(self, unknown_label="__UNKNOWN__"):
        super().__init__()
        self.unknown_label = unknown_label
        self.unknown_index = None

    def fit(self, y):
        # Fit the normal label encoder
        super().fit(list(y) + [self.unknown_label])
        # Store the index for the unknown label
        self.unknown_index = self.transform([self.unknown_label])[0]
        return self
    
    def transform(self, y):
        # For any label not in classes_, map to unknown_index
        y_encoded = []
        for item in y:
            if item in self.classes_:
                y_encoded.append(super().transform([item])[0])
            else:
                y_encoded.append(self.unknown_index)
        return y_encoded


def main():
    # ---------------------------------------------------------------------
    # 1. LOAD & RESHAPE DATA
    # ---------------------------------------------------------------------
    # Reads the wide CSV, which has columns like:
    # Year, Sport, AIN, ALG, ANZ, ARG, ARM, AUS, ...
    df_wide = pd.read_csv('C:/Users/leonhuangzekai/Desktop/MCM/MCM-2025/updated_gold_medals_int.csv')
    
    # Melt from wide to long format:
    #   [Year, Sport, Country, GoldMedals]
    df_long = df_wide.melt(
        id_vars=["Year", "Sport"], 
        var_name="Country", 
        value_name="GoldMedals"
    )
    
    # ---------------------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # ---------------------------------------------------------------------
    # Sort so rolling/lag features make sense
    df_long = df_long.sort_values(["Country", "Sport", "Year"]).reset_index(drop=True)
    
    # Example rolling feature: average medals in the previous 2 Olympics
    # We'll group by (Country, Sport), shift by 1 to not peek at current,
    # and roll the last 2. For years with fewer than 2 past results, it’ll just use what’s available.
    df_long["RollingMeanPrev2"] = df_long.groupby(["Country", "Sport"])["GoldMedals"] \
                                         .transform(lambda x: x.shift().rolling(2, min_periods=1).mean())
    
    # We can fill any remaining NaN (for the first row(s)) with 0
    df_long["RollingMeanPrev2"] = df_long["RollingMeanPrev2"].fillna(0)
    
    # You could add more features here (GDP, population, etc.) if you have them.
    
    # ---------------------------------------------------------------------
    # 3. SPLIT DATA INTO TRAIN & TEST BASED ON YEAR
    # ---------------------------------------------------------------------
    # Let's say we train on all data up to year 2012, and test on 2016 & beyond (if present).
    # Adjust these thresholds to match your dataset’s coverage.
    train_df = df_long[df_long["Year"] <= 2012].copy()
    test_df  = df_long[df_long["Year"]  > 2012].copy()
    
    # If you want to predict specifically for 2024 or 2028, you can
    # subset test_df to only those years once you have real data for
    # your other predictive features, or keep them all for demonstration.
    
    # ---------------------------------------------------------------------
    # 4. PREPARE FEATURES & TARGET
    # ---------------------------------------------------------------------
    # We'll treat "GoldMedals" as the numeric target. Because gold medals
    # are typically small integers (0,1,...), let's do a log transform:
    #    y = log(1 + GoldMedals)
    # This often helps a model handle the skew.
    
    train_df["Target"] = np.log1p(train_df["GoldMedals"])
    test_df["Target"]  = np.log1p(test_df["GoldMedals"])
    
    # Encode "Country" and "Sport" as numeric labels (simple approach).
    # Alternatively, you can use one-hot encoding or XGBoost’s built-in
    # categorical handling (with latest versions).
    # Fit only on train data (plus an artificial UNKNOWN token)
    country_le = SafeLabelEncoder()
    country_le.fit(train_df["Country"])

    sport_le = SafeLabelEncoder()
    sport_le.fit(train_df["Sport"])

    # Transform train
    train_df["CountryEnc"] = country_le.transform(train_df["Country"])
    train_df["SportEnc"]   = sport_le.transform(train_df["Sport"])

    # Transform test (unseen labels become the 'unknown' index)
    test_df["CountryEnc"] = country_le.transform(test_df["Country"])
    test_df["SportEnc"]   = sport_le.transform(test_df["Sport"])

    
    # Now define the feature columns we want to use for the model
    feature_cols = ["Year", "CountryEnc", "SportEnc", "RollingMeanPrev2"]
    
    X_train = train_df[feature_cols]
    y_train = train_df["Target"]  # log(1 + GoldMedals)
    
    X_test  = test_df[feature_cols]
    y_test  = test_df["Target"]   # log(1 + GoldMedals)
    
    # ---------------------------------------------------------------------
    # 5. TRAIN MODEL (XGBoost Regressor)
    # ---------------------------------------------------------------------
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # ---------------------------------------------------------------------
    # 6. EVALUATE ON TEST
    # ---------------------------------------------------------------------
    # Predict (log scale)
    pred_log = model.predict(X_test)
    # Convert back from log scale: exp(pred_log) - 1
    pred_medals = np.expm1(pred_log)
    
    # Compare to actual medals in test set
    actual_medals = np.expm1(y_test)
    
    # Let's compute RMSE in "actual medal" space
    rmse = mean_squared_error(actual_medals, pred_medals, squared=False)
    print(f"Test RMSE (on actual medals): {rmse:.4f}")
    
    # ---------------------------------------------------------------------
    # 7. MAKE FUTURE PREDICTIONS (EXAMPLE)
    # ---------------------------------------------------------------------
    # If you have a future year (say 2024) with the needed features
    # (Year=2024, RollingMeanPrev2, etc.), you can feed that in here.
    # For demo, let's just show how you'd create a "future_df" and predict.
    
    # Suppose we want to predict 2024 for every (Country, Sport). 
    # Realistically, you'd merge or fill the "RollingMeanPrev2" from
    # the latest available data in 2020 or 2021, etc.
    # For demonstration, let's do a simplistic approach:
    
    # 1) Start with a unique combination of country & sport from your data
    unique_country_sport = df_long[["Country", "Sport"]].drop_duplicates()
    
    # 2) Assign them all the same Year=2024 (or 2028, etc.)
    future_df = unique_country_sport.copy()
    future_df["Year"] = 2024
    
    # 3) For RollingMeanPrev2, you’d ideally compute it from the
    #    most recent Olympic results (e.g. 2016, 2020). For a toy example:
    #    Let’s just reuse the RollingMeanPrev2 from 2012 or 2016. 
    #    In a real project, you'd do more precise logic here.
    
    # We'll do a left-join from a recent year’s data to get approximate RollingMeanPrev2
    # For simplicity, pick the last known year in train_df, e.g. 2012:
    last_known = train_df[train_df["Year"] == 2012][["Country","Sport","RollingMeanPrev2"]]
    
    # Merge on (Country,Sport) 
    future_df = future_df.merge(last_known, on=["Country","Sport"], how="left")
    future_df["RollingMeanPrev2"] = future_df["RollingMeanPrev2"].fillna(0)
    
    # Encode future data
    future_df["CountryEnc"] = country_le.transform(future_df["Country"])
    future_df["SportEnc"]   = sport_le.transform(future_df["Sport"])
    
    # Build the feature matrix for the future
    future_X = future_df[["Year", "CountryEnc", "SportEnc", "RollingMeanPrev2"]]
    
    # Predict log gold medals
    future_pred_log = model.predict(future_X)
    future_pred_medals = np.expm1(future_pred_log)
    
    future_df["PredictedGoldMedals"] = future_pred_medals
    
    print("\nSample Future Predictions (Year=2024):")
    print(future_df.head(10))
    
    # You could then group by Country to see total predicted golds, etc.
    # For example:
    country_sums = future_df.groupby("Country")["PredictedGoldMedals"].sum().reset_index()
    country_sums = country_sums.sort_values("PredictedGoldMedals", ascending=False)
    print("\nPredicted total golds by country in 2024 (top 10):")
    print(country_sums.head(10))

if __name__ == "__main__":
    main()
