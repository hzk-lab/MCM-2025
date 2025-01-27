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
    假设 df 里包含 target_year (哪怕是占位行).
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
    对 (Sport, Country) 做 label encoding, 并用 log(1 + GoldMedals) 作为 y.
    如果 GoldMedals 为 NaN => y 为 NaN.
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
    y_true_log, y_pred_log 都是 log(1+GoldMedals).
    """
    if y_true_log.isna().any():
        # 若真实值是 NaN, 无法算 RMSE
        return np.nan
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return mean_squared_error(y_true, y_pred, squared=False)

def train_and_eval_params(X_train, y_train, X_test, y_test, params):
    """
    给定一组 XGBoost 超参数, 在训练集训练, 在测试集计算 RMSE.
    如果测试集全是 NaN 无法计算, 返回 np.nan.
    """
    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    rmse = calc_rmse_in_original_space(y_test, y_pred_log)
    return rmse

def main():
    # ========== 1. 读取“宽表”CSV + melt成长表 ==========
    df_wide = pd.read_csv('C:/Users/leonhuangzekai/Desktop/MCM/updated_bronze_medals_int.csv')
    df_long = df_wide.melt(
        id_vars=["Year", "Sport"],
        var_name="Country",
        value_name="GoldMedals"
    )

    # ========== 2. 插入 2028 占位行, 方便后面统一处理 ==========
    all_country_sport = df_long[["Country","Sport"]].drop_duplicates()
    future_year = 2028
    future_data = all_country_sport.copy()
    future_data["Year"] = future_year
    future_data["GoldMedals"] = np.nan  # 无真实值
    df_long_extended = pd.concat([df_long, future_data], ignore_index=True)

    # ========== 3. 滚动特征 & Label Encode ==========
    df_long_extended = create_rolling_features(df_long_extended)
    sport_le = LabelEncoder()
    country_le = LabelEncoder()
    sport_le.fit(df_long_extended["Sport"])
    country_le.fit(df_long_extended["Country"])

    # ========== 4. 定义多窗口预测的目标年份(用于评估 & 选 k) ==========
    # 这里假设 [2012, 2016, 2020, 2024] 都有真实数据可以算RMSE
    target_years_for_search = [2012, 2016, 2020, 2024]

    # 超参数搜索空间 (这里先固定, 你也可再套一层搜索)
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
    
    # ========== 5. 在 k ∈ [1..10] 中搜索最佳 k ==========
    k_candidates = range(1, 11)  # 1~10
    best_k = None
    best_k_rmse = float("inf")
    best_k_params_record = []  # 用于记录不同 k 的平均RMSE
    
    for k in k_candidates:
        # 对每个 k, 我们也要像之前那样对多个年份做滚动预测+超参搜索，
        # 再把这些年份的RMSE做平均 => 得到对该 k 的评价。
        # -----------
        # 不同写法：你可以先对 param 做搜索，也可以先对 k 做搜索。
        # 这里示例：对 param 做网格搜索。
        
        # 累加这个 k 对应的多个 target_year 的 RMSE
        rmse_list_for_this_k = []
        
        for tyear in target_years_for_search:
            # 先切出 train_df, test_df
            train_df, test_df = make_sliding_window_data(df_long_extended, tyear, k)
            X_train, y_train = prepare_features_and_labels(train_df, sport_le, country_le)
            X_test,  y_test  = prepare_features_and_labels(test_df,  sport_le, country_le)

            # 在这个 (X_train, y_train) + (X_test, y_test) 上找最优超参数
            best_rmse = float("inf")
            for params in candidate_params:
                rmse = train_and_eval_params(X_train, y_train, X_test, y_test, params)
                if not np.isnan(rmse) and rmse < best_rmse:
                    best_rmse = rmse
            if best_rmse == float("inf"):
                # 说明无法计算 RMSE (可能没真实值)
                # 这里可以忽略 或 赋值一个 np.nan
                best_rmse = np.nan

            rmse_list_for_this_k.append(best_rmse)

        # 对 k 的综合表现(平均RMSE)
        # 如果某些窗口 RMSE=NaN, 可选择只在非NaN上做平均 or 算 overall NaN
        rmse_array = np.array(rmse_list_for_this_k)
        valid_rmse = rmse_array[~np.isnan(rmse_array)]
        if len(valid_rmse) == 0:
            # 说明所有窗口都没法算 RMSE, 那就跳过
            avg_rmse = np.nan
        else:
            avg_rmse = valid_rmse.mean()

        best_k_params_record.append((k, avg_rmse))
        print(f"[k={k}] average RMSE across {target_years_for_search} = {avg_rmse}")
        
        # 是否更新最优
        if not np.isnan(avg_rmse) and avg_rmse < best_k_rmse:
            best_k_rmse = avg_rmse
            best_k = k

    print("\n=== Summary of k search ===")
    for (kval, score) in best_k_params_record:
        print(f"k={kval}, mean RMSE={score}")

    print(f"\nBest k = {best_k}, with average RMSE={best_k_rmse}")
    
    if best_k is None:
        print("无法选到合适的 k，可能没有足够的真实数据。停止。")
        return

    # ========== 6. 决定最终 (k, 超参数) 并预测 2028 ==========
    #
    # 这里为了简化：选出 best_k 后，我们再在 [2012, 2016, 2020, 2024] 上搜一遍超参数，
    # 取该 k 下最好的超参数(对多窗口再做一次融合)。
    # 当然你也可以在上面的循环里把最优param一起记录下来，再整合。
    # 下面做一个简单的方式：对 param 做“多窗口平均RMSE”再取最优。

    candidate_k_params = []
    for param in candidate_params:
        # 对 param 做多窗口平均
        rmse_list = []
        for tyear in target_years_for_search:
            train_df, test_df = make_sliding_window_data(df_long_extended, tyear, best_k)
            X_train, y_train = prepare_features_and_labels(train_df, sport_le, country_le)
            X_test,  y_test  = prepare_features_and_labels(test_df,  sport_le, country_le)
            
            rmse = train_and_eval_params(X_train, y_train, X_test, y_test, param)
            if not np.isnan(rmse):
                rmse_list.append(rmse)
        
        # 计算param在所有可评估窗口的平均rmse
        if len(rmse_list) > 0:
            avg_rmse = np.mean(rmse_list)
        else:
            avg_rmse = np.nan
        
        candidate_k_params.append((param, avg_rmse))
    
    # 选出平均rmse最低的param
    valid_candidates = [cp for cp in candidate_k_params if not np.isnan(cp[1])]
    if len(valid_candidates)==0:
        print("在 best_k 下无法找到可评估的超参数, 无法继续预测。")
        return

    best_param, best_param_rmse = min(valid_candidates, key=lambda x: x[1])
    print(f"\nBest param under k={best_k} is {best_param}, mean RMSE={best_param_rmse}")

    # ========== 7. 用最终 (k, best_param) 做对 2028 的预测 ==========
    predict_year = 2028
    # 训练集 = 前 k 届
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

    # 保存
    results_2028.to_csv("final_prediction_2028_with_best_k_bronze.csv", index=False)
    print(f"\nDone. Best k={best_k}, param={best_param}.\n2028 prediction saved to 'final_prediction_2028_with_best_k.csv'.")

if __name__ == "__main__":
    main()
