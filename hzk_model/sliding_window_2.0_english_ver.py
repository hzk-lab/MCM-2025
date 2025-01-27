#!/usr/bin/env python3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import collections

def create_rolling_features(df):
    """
    Create RollingMeanPrev2 for each row: (Country, Sport) calculates the 
    average number of gold medals in the previous 2 Olympics.
    """
    df = df.sort_values(["Country", "Sport", "Year"]).reset_index(drop=True)
    df["RollingMeanPrev2"] = (
        df.groupby(["Country", "Sport"])["GoldMedals"]
          .transform(lambda x: x.shift().rolling(2, min_periods=1).mean())
    )
    df["RollingMeanPrev2"] = df["RollingMeanPrev2"].fillna(0)
    return df

def make_sliding_window_data(df, target_year, k):
    """
    Construct data where the previous k Olympics are used as training data 
    and 'target_year' is used as test data.
    Assumes that 'target_year' is included in the dataset (even as a placeholder row).
    """
    all_years = sorted(df["Year"].unique())
    if target_year not in all_years:
        raise ValueError(f"Target year {target_year} not found in dataset.")
    
    idx = all_years.index(target_year)
    start_idx = max(0, idx - k)
    train_years = all_years[start_idx:idx]

    train_df = df[df["Year"].isin(train_years)].copy()
    test_df  = df[df["Year"] == target_year].copy()
    return train_df, test_df

def prepare_features_and_labels(df, sport_le, country_le):
    """
    Perform label encoding for (Sport, Country) and use log(1 + GoldMedals) as the target (y).
    If GoldMedals is NaN, y will also be NaN.
    """
    df = df.sort_values(["Country", "Sport", "Year"]).reset_index(drop=True)
    df["Target"] = np.log1p(df["GoldMedals"])  # NaN -> NaN

    df["SportEnc"]   = sport_le.transform(df["Sport"])
    df["CountryEnc"] = country_le.transform(df["Country"])

    feature_cols = ["Year", "SportEnc", "CountryEnc", "RollingMeanPrev2"]
    X = df[feature_cols].copy()
    y = df["Target"].copy()
    return X, y

def calc_rmse_in_original_space(y_true_log, y_pred_log):
    """
    Calculate RMSE in the original gold medal space.
    y_true_log and y_pred_log are log(1 + GoldMedals).
    """
    if y_true_log.isna().any():
        # If the true value is NaN, RMSE cannot be calculated
        return np.nan
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return mean_squared_error(y_true, y_pred, squared=False)

def train_and_eval_params(X_train, y_train, X_test, y_test, params):
    """
    Given a set of XGBoost hyperparameters, train on the training set 
    and calculate RMSE on the test set.
    If the test set contains all NaN values and RMSE cannot be calculated, return np.nan.
    """
    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    rmse = calc_rmse_in_original_space(y_test, y_pred_log)
    return rmse

def main():
    # ========== 1. Read the "wide table" CSV and melt it into a long table ==========
    df_wide = pd.read_csv('C:/Users/leonhuangzekai/Desktop/MCM/updated_total_medals_int.csv')
    df_long = df_wide.melt(
        id_vars=["Year", "Sport"],
        var_name="Country",
        value_name="GoldMedals"
    )

    # ========== 2. Insert a placeholder row for 2028 to handle uniformly later ==========
    all_country_sport = df_long[["Country","Sport"]].drop_duplicates()
    future_year = 2028
    future_data = all_country_sport.copy()
    future_data["Year"] = future_year
    future_data["GoldMedals"] = np.nan  # No actual values
    df_long_extended = pd.concat([df_long, future_data], ignore_index=True)

    # ========== 3. Create rolling features and perform label encoding ==========
    df_long_extended = create_rolling_features(df_long_extended)
    sport_le = LabelEncoder()
    country_le = LabelEncoder()
    sport_le.fit(df_long_extended["Sport"])
    country_le.fit(df_long_extended["Country"])

    # ========== 4. Define target years for multi-window prediction (for evaluation and k selection) ==========
    # Assume that [2012, 2016, 2020, 2024] all have actual data for RMSE calculation
    target_years_for_search = [2012, 2016, 2020, 2024]

    # Hyperparameter search space (fixed here, you can further expand it)
    candidate_params = [
        {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 100},
        {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 150},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 100},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 150},
        {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100},
        {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 150},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 100},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 150},
    ]
    
    # ========== 5. Search for the best k ∈ [1..10] ==========
    k_candidates = range(1, 11)  # 1~10
    best_k = None
    best_k_rmse = float("inf")
    best_k_params_record = []  # Record the average RMSE for different k values
    
    for k in k_candidates:
        # For each k, we also need to make rolling predictions + hyperparameter search 
        # across multiple target years, then average these RMSEs to evaluate k.
        
        rmse_list_for_this_k = []
        
        for tyear in target_years_for_search:
            # Extract train_df and test_df
            train_df, test_df = make_sliding_window_data(df_long_extended, tyear, k)
            X_train, y_train = prepare_features_and_labels(train_df, sport_le, country_le)
            X_test,  y_test  = prepare_features_and_labels(test_df,  sport_le, country_le)

            # Search for the best hyperparameters on this (X_train, y_train) + (X_test, y_test)
            best_rmse = float("inf")
            for params in candidate_params:
                rmse = train_and_eval_params(X_train, y_train, X_test, y_test, params)
                if not np.isnan(rmse) and rmse < best_rmse:
                    best_rmse = rmse
            if best_rmse == float("inf"):
                # If RMSE cannot be calculated, assign np.nan
                best_rmse = np.nan

            rmse_list_for_this_k.append(best_rmse)

        # Calculate the overall performance (average RMSE) for k
        # If some windows have NaN RMSE, choose to average only non-NaN or assign overall NaN
        rmse_array = np.array(rmse_list_for_this_k)
        valid_rmse = rmse_array[~np.isnan(rmse_array)]
        if len(valid_rmse) == 0:
            # If no RMSE can be calculated, skip
            avg_rmse = np.nan
        else:
            avg_rmse = valid_rmse.mean()

        best_k_params_record.append((k, avg_rmse))
        print(f"[k={k}] average RMSE across {target_years_for_search} = {avg_rmse}")
        
        # Update best k
        if not np.isnan(avg_rmse) and avg_rmse < best_k_rmse:
            best_k_rmse = avg_rmse
            best_k = k

    print("\n=== Summary of k search ===")
    for (kval, score) in best_k_params_record:
        print(f"k={kval}, mean RMSE={score}")

    print(f"\nBest k = {best_k}, with average RMSE={best_k_rmse}")
    
    if best_k is None:
        print("No suitable k was found; there may not be enough actual data. Stopping.")
        return

    # ========== 6. Determine final (k, hyperparameters) and predict for 2028 ==========
    # To simplify: after selecting best_k, search hyperparameters again across [2012, 2016, 2020, 2024]
    # and use the best hyperparameters for this k to make predictions for 2028.
    # This is a simpler method: calculate "average RMSE for multiple windows" for hyperparameters.
    
    candidate_k_params = []
    for param in candidate_params:
        # Perform multi-window averaging for this param
        rmse_list = []
        for tyear in target_years_for_search:
            train_df, test_df = make_sliding_window_data(df_long_extended, tyear, best_k)
            X_train, y_train = prepare_features_and_labels(train_df, sport_le, country_le)
            X_test,  y_test  = prepare_features_and_labels(test_df,  sport_le, country_le)
            
            rmse = train_and_eval_params(X_train, y_train, X_test, y_test, param)
            if not np.isnan(rmse):
                rmse_list.append(rmse)
        
        # Calculate the average RMSE of param across all evaluable windows
        if len(rmse_list) > 0:
            avg_rmse = np.mean(rmse_list)
        else:
            avg_rmse = np.nan
        
        candidate_k_params.append((param, avg_rmse))
    
    # Select the param with the lowest average RMSE
    valid_candidates = [cp for cp in candidate_k_params if not np.isnan(cp[1])]
    if len(valid_candidates)==0:
        print("No evaluable hyperparameters were found for best_k, cannot proceed with prediction.")
        return

    best_param, best_param_rmse = min(valid_candidates, key=lambda x: x[1])
    print(f"\nBest param under k={best_k} is {best_param}, mean RMSE={best_param_rmse}")

    # ========== 7. Use final (k, best_param) to predict for 2028 ==========
    predict_year = 2028
    # Training set = previous k Olympics
    train_df_2028, test_df_2028 = make_sliding_window_data(df_long_extended, predict_year, best_k)
    X_train_2028, y_train_2028 = prepare_features_and_labels(train_df_2028, sport_le, country_le)
    X_test_2028,  y_test_2028  = prepare_features_and_labels(test_df_2028,  sport_le, country_le)

    final_model = xgb.XGBRegressor(**best_param, random_state=42)
    final_model.fit(X_train_2028, y_train_2028)

    pred_log = final_model.predict(X_test_2028)
    pred_gold = np.expm1(pred_log)

    results_2028 = test_df_2028[["Year","Sport","Country","GoldMedals"]].copy()
    results_2028.rename(columns={"GoldMedals":"ActualGold"}, inplace=True)
    results_2028["PredictedGold"] = pred_gold

    # Save the results
    results_2028.to_csv("final_prediction_2028_with_best_k_total.csv", index=False)
    print(f"\nDone. Best k={best_k}, param={best_param}.\n2028 prediction saved to 'final_prediction_2028_with_best_k.csv'.")

if __name__ == "__main__":
    main()
