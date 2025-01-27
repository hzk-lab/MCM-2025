#!/usr/bin/env python3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import collections

def create_rolling_features(df):
    """
    为每行创建 RollingMeanPrev2: (Country, Sport) 在之前 2 届的平均金牌数.
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
    构造 “前 k 届” 作为 train, “target_year” 作为 test.
    如果 target_year 不在 df 中, 会抛错. 
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
    对 (Sport, Country) 做 label encoding, 用 log(1 + GoldMedals) 作为 y.
    如果 GoldMedals 为 NaN => y 也为 NaN.
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
    在原始金牌空间里计算RMSE.
    """
    if y_true_log.isna().any():
        return np.nan  # 真实值为空, 无法算RMSE
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return mean_squared_error(y_true, y_pred, squared=False)

def train_and_eval_params(X_train, y_train, X_test, y_test, params):
    """
    在训练集训练, 在测试集计算 RMSE (若无法计算则 NaN).
    """
    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    rmse = calc_rmse_in_original_space(y_test, y_pred_log)
    return rmse

def main():
    # 1. 读原始 CSV + melt => 长表
    df_wide = pd.read_csv("updated_gold_medals_int.csv")
    df_long = df_wide.melt(
        id_vars=["Year", "Sport"],
        var_name="Country",
        value_name="GoldMedals"
    )

    # 2. 忽略 2024 的真实数据 => 将 Year=2024 的 GoldMedals 置为 NaN
    df_long.loc[df_long["Year"] == 2024, "GoldMedals"] = np.nan

    # 3. 插入 2028 也当成未来(占位行: GoldMedals=NaN)
    all_cs = df_long[["Country","Sport"]].drop_duplicates()
    df_2028 = all_cs.copy()
    df_2028["Year"] = 2028
    df_2028["GoldMedals"] = np.nan
    df_long_extended = pd.concat([df_long, df_2028], ignore_index=True)

    # 4. 构造滚动特征 + LabelEncoding
    df_long_extended = create_rolling_features(df_long_extended)

    sport_le = LabelEncoder()
    country_le = LabelEncoder()
    sport_le.fit(df_long_extended["Sport"])
    country_le.fit(df_long_extended["Country"])

    # 5. 定义用于评估 & 选 k 的年份. 
    #    因为我们“废弃”了 2024 的真实数据, 所以 2024 不能再作为评估目标.
    #    例如只用 [2012, 2016, 2020] (这些还保留真实数据).
    target_years_for_search = [2012, 2016, 2020]

    # 超参数候选
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
    
    # ========== 6. 我们把 k 也当做超参数来搜 1~8 (随意定义) ==========
    k_candidates = range(1, 9)
    best_k = None
    best_k_rmse = float("inf")

    # 存下每个 k 的平均 RMSE
    k_score_list = []

    for k in k_candidates:
        # 对本 k, 在 [2012, 2016, 2020] 这些年进行滚动窗口+超参搜索
        # 求得“多窗口平均RMSE” -> 评估这个 k 的好坏
        window_rmses = []
        for year in target_years_for_search:
            train_df, test_df = make_sliding_window_data(df_long_extended, year, k)
            X_train, y_train = prepare_features_and_labels(train_df, sport_le, country_le)
            X_test,  y_test  = prepare_features_and_labels(test_df,  sport_le, country_le)
            
            # 找该窗口(年)下最优超参数
            best_rmse_for_window = float("inf")
            for p in candidate_params:
                rmse = train_and_eval_params(X_train, y_train, X_test, y_test, p)
                if rmse is not None and not np.isnan(rmse) and rmse < best_rmse_for_window:
                    best_rmse_for_window = rmse
            if best_rmse_for_window == float("inf"):
                # 无法计算RMSE
                best_rmse_for_window = np.nan
            
            window_rmses.append(best_rmse_for_window)

        # 对 k 的整体表现(平均RMSE)
        rmse_array = np.array(window_rmses)
        valid_rmse = rmse_array[~np.isnan(rmse_array)]
        if len(valid_rmse)==0:
            avg_rmse = np.nan
        else:
            avg_rmse = valid_rmse.mean()

        k_score_list.append((k, avg_rmse))
        if not np.isnan(avg_rmse) and avg_rmse < best_k_rmse:
            best_k_rmse = avg_rmse
            best_k = k

    print("\n===== K search result =====")
    for (kval, score) in k_score_list:
        print(f"K={kval}, MeanRMSE={score}")
    print(f"Best K={best_k}, with average RMSE={best_k_rmse}")

    if best_k is None:
        print("No valid k found. Stop.")
        return

    # ========== 7. 确定对 best_k 下再选最佳超参数 (多窗口平均) ==========
    param_scores = []
    for p in candidate_params:
        # 计算 p 在 [2012, 2016, 2020] 下的平均RMSE
        p_rmse_list = []
        for year in target_years_for_search:
            train_df, test_df = make_sliding_window_data(df_long_extended, year, best_k)
            X_train, y_train = prepare_features_and_labels(train_df, sport_le, country_le)
            X_test,  y_test  = prepare_features_and_labels(test_df,  sport_le, country_le)

            rmse = train_and_eval_params(X_train, y_train, X_test, y_test, p)
            if not np.isnan(rmse):
                p_rmse_list.append(rmse)
        
        if len(p_rmse_list)>0:
            mean_rmse = np.mean(p_rmse_list)
        else:
            mean_rmse = np.nan

        param_scores.append((p, mean_rmse))

    valid_params = [(pp,rr) for (pp,rr) in param_scores if not np.isnan(rr)]
    if len(valid_params)==0:
        print("No valid param found with best_k. Stop.")
        return

    best_param, best_param_rmse = min(valid_params, key=lambda x: x[1])
    print(f"\nBest param with k={best_k} => {best_param}, meanRMSE={best_param_rmse}")

    # ========== 8. 最终对 2024 预测 (不含真实数据) ==========
    # 现在 2024 的 GoldMedals 全是NaN => 只能得到 PredictedGold, 无法计算RMSE
    predict_year = 2024
    train_df_2024, test_df_2024 = make_sliding_window_data(df_long_extended, predict_year, best_k)
    X_train_2024, y_train_2024 = prepare_features_and_labels(train_df_2024, sport_le, country_le)
    X_test_2024,  y_test_2024  = prepare_features_and_labels(test_df_2024,  sport_le, country_le)

    final_model_2024 = xgb.XGBRegressor(**best_param, random_state=42)
    final_model_2024.fit(X_train_2024, y_train_2024)
    pred_log_2024 = final_model_2024.predict(X_test_2024)
    pred_gold_2024 = np.expm1(pred_log_2024)

    df_2024_result = test_df_2024[["Year","Sport","Country","GoldMedals"]].copy()
    df_2024_result.rename(columns={"GoldMedals":"ActualGold"}, inplace=True)
    df_2024_result["PredictedGold"] = pred_gold_2024
    df_2024_result.to_csv("prediction_2024_ignored_real.csv", index=False)
    print("\nPredictions for 2024 saved to 'prediction_2024_ignored_real.csv'.")
'''
    # (可选) 9. 对 2028 做同样预测
    # 如果想对 2028 也输出, 不妨再做一遍:
    train_df_2028, test_df_2028 = make_sliding_window_data(df_long_extended, 2028, best_k)
    X_train_2028, y_train_2028 = prepare_features_and_labels(train_df_2028, sport_le, country_le)
    X_test_2028,  y_test_2028  = prepare_features_and_labels(test_df_2028,  sport_le, country_le)

    final_model_2028 = xgb.XGBRegressor(**best_param, random_state=42)
    final_model_2028.fit(X_train_2028, y_train_2028)
    pred_log_2028 = final_model_2028.predict(X_test_2028)
    pred_gold_2028 = np.expm1(pred_log_2028)

    df_2028_result = test_df_2028[["Year","Sport","Country","GoldMedals"]].copy()
    df_2028_result.rename(columns={"GoldMedals":"ActualGold"}, inplace=True)
    df_2028_result["PredictedGold"] = pred_gold_2028
    df_2028_result.to_csv("prediction_2028_ignored_real.csv", index=False)
    print("\nPredictions for 2028 saved to 'prediction_2028_ignored_real.csv'.")
'''
if __name__ == "__main__":
    main()
