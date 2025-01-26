#!/usr/bin/env python3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import collections

def create_rolling_features(df):
    """
    为每行创建一个 RollingMeanPrev2:
    表示 (Country, Sport) 在之前 2 届奥运会的平均金牌数.
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
    给定目标年份 target_year、窗口大小 k，返回:
      - train_df: 目标年前的 k 届数据
      - test_df:  目标年（Year == target_year）的数据

    如果 df 当中不含 target_year，会报错；
    不过我们会在主流程里先插入 2028 的占位行.
    """
    all_years = sorted(df["Year"].unique())
    if target_year not in all_years:
        raise ValueError(f"Target year {target_year} not found in dataset.")
    
    target_idx = all_years.index(target_year)
    start_idx = max(0, target_idx - k)
    train_years = all_years[start_idx:target_idx]  # 不含 target_year

    train_df = df[df["Year"].isin(train_years)].copy()
    test_df  = df[df["Year"] == target_year].copy()
    return train_df, test_df

def prepare_features_and_labels(df, sport_le, country_le):
    """
    把 (Sport, Country) 做 LabelEncode, 并生成特征 X, 标签 y.
    y = log(1 + GoldMedals).

    如果 GoldMedals 是 NaN (未来预测情况), 那么 y 也会是 NaN.
    """
    df = df.sort_values(["Country", "Sport", "Year"]).reset_index(drop=True)

    # 目标: log(1 + GoldMedals)
    # 如果 GoldMedals 为 NaN，log1p 会是 NaN
    df["Target"] = np.log1p(df["GoldMedals"])

    df["SportEnc"]   = sport_le.transform(df["Sport"])
    df["CountryEnc"] = country_le.transform(df["Country"])

    feature_cols = ["Year", "SportEnc", "CountryEnc", "RollingMeanPrev2"]
    X = df[feature_cols].copy()
    y = df["Target"].copy()
    return X, y

def calc_rmse_in_original_space(y_true_log, y_pred_log):
    """
    在原始金牌空间下计算RMSE。
    y_true_log, y_pred_log 如果为 log(1+GoldMedals),
    先还原再做 MSE.
    """
    # 如果有 NaN，会导致 RMSE 无法计算，需要小心
    if y_true_log.isna().any():
        return np.nan

    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return mean_squared_error(y_true, y_pred, squared=False)

def train_and_eval_params(X_train, y_train, X_test, y_test, param):
    """
    用给定的超参数 param 训练 XGBoost, 然后在 (X_test, y_test) 上
    计算 RMSE (原始金牌空间).
    如果 y_test 都是 NaN (没有真实数据), 返回 NaN.
    """
    model = xgb.XGBRegressor(
        max_depth=param["max_depth"],
        learning_rate=param["learning_rate"],
        n_estimators=param["n_estimators"],
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    rmse = calc_rmse_in_original_space(y_test, y_pred_log)
    return rmse

def main():
    # ========== 1. 读取“宽表”CSV并转换成长表 ==========
    df_wide = pd.read_csv('C:/Users/leonhuangzekai/Desktop/MCM/MCM-2025/updated_gold_medals_int.csv')
    # 宽 -> 长
    df_long = df_wide.melt(
        id_vars=["Year", "Sport"],
        var_name="Country",
        value_name="GoldMedals"
    )

    # ========== 2. 在 df_long 中插入 2028 占位行 (GoldMedals=NaN) ==========
    #    这样 make_sliding_window_data(..., 2028, k) 才不会报错“找不到 2028”
    future_year = 2028
    all_country_sport = df_long[["Country", "Sport"]].drop_duplicates()
    future_data = all_country_sport.copy()
    future_data["Year"] = future_year
    future_data["GoldMedals"] = np.nan  # 无法知道真实金牌数

    # 把 future_data 拼接到 df_long
    df_long_extended = pd.concat([df_long, future_data], ignore_index=True)

    # ========== 3. 构造滚动特征 RollingMeanPrev2 ==========
    df_long_extended = create_rolling_features(df_long_extended)

    # ========== 4. LabelEncoder (在全量数据上 fit) ==========
    sport_le = LabelEncoder()
    country_le = LabelEncoder()
    sport_le.fit(df_long_extended["Sport"])
    country_le.fit(df_long_extended["Country"])

    # ========== 5. 多窗口超参数搜索 (针对过去若干年) ==========
    # 这里假设我们有真实数据的最近年份有 [2012, 2016, 2020, 2024]。
    # 先做滚动窗口预测+调参 (k=3) 来选出若干窗口的“局部最优参数”。
    target_years_for_search = [2012, 2016, 2020, 2024]
    k = 3

    # 简易的超参网格
    param_grid = {
        "max_depth": [4, 6],
        "learning_rate": [0.1, 0.05],
        "n_estimators": [100, 150]
    }
    candidate_params = []
    for md in param_grid["max_depth"]:
        for lr in param_grid["learning_rate"]:
            for ne in param_grid["n_estimators"]:
                candidate_params.append({
                    "max_depth": md,
                    "learning_rate": lr,
                    "n_estimators": ne
                })

    best_params_per_window = []

    for tyear in target_years_for_search:
        # 构造 train_df, test_df
        train_df, test_df = make_sliding_window_data(df_long_extended, tyear, k)
        X_train, y_train = prepare_features_and_labels(train_df, sport_le, country_le)
        X_test,  y_test  = prepare_features_and_labels(test_df,  sport_le, country_le)

        best_rmse = float("inf")
        best_param = None

        for param in candidate_params:
            rmse = train_and_eval_params(X_train, y_train, X_test, y_test, param)
            # 如果 test_df 的金牌全是 NaN, rmse 会是 NaN, 不能比较
            if rmse is not None and not np.isnan(rmse):
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_param = param

        if best_param is None:
            # 如果完全没有真实数据, 那就没法选参
            print(f"[{tyear}] No real medals available, can't pick best param.")
        else:
            print(f"[{tyear}] Best Param={best_param}, RMSE={best_rmse:.4f}")
            best_params_per_window.append(best_param)

    # ========== 6. 汇总多个窗口的最优超参数 ==========
    #    例: 对 max_depth, n_estimators 取众数, 对 learning_rate 取平均。
    if len(best_params_per_window) == 0:
        print("No best params found from any window! Perhaps no real data is available.")
        return

    max_depth_list = [bp["max_depth"] for bp in best_params_per_window]
    lr_list = [bp["learning_rate"] for bp in best_params_per_window]
    n_est_list = [bp["n_estimators"] for bp in best_params_per_window]

    # 频率最高的 max_depth
    max_depth_final = collections.Counter(max_depth_list).most_common(1)[0][0]
    # 平均 learning_rate
    lr_final = sum(lr_list) / len(lr_list)
    # 频率最高的 n_estimators
    n_est_final = collections.Counter(n_est_list).most_common(1)[0][0]

    final_params = {
        "max_depth": max_depth_final,
        "learning_rate": lr_final,
        "n_estimators": n_est_final
    }
    print("\n===> Final consolidated params:", final_params)

    # ========== 7. 用最终参数预测 2028 ==========
    #    根据需求决定训练集：可以还是“前 k 届” (即 2016,2020,2024)；
    #    也可以把全部历史(或更多年)都拿来。
    #    这里示例还是用 k=3, 所以拿 [2024, 2020, 2016] 做训练。
    predict_year = 2028

    train_df_2028, test_df_2028 = make_sliding_window_data(df_long_extended, predict_year, k)
    X_train_2028, y_train_2028 = prepare_features_and_labels(train_df_2028, sport_le, country_le)
    X_test_2028,  y_test_2028  = prepare_features_and_labels(test_df_2028,  sport_le, country_le)

    # 训练最终模型
    final_model = xgb.XGBRegressor(
        max_depth=final_params["max_depth"],
        learning_rate=final_params["learning_rate"],
        n_estimators=final_params["n_estimators"],
        random_state=42
    )
    final_model.fit(X_train_2028, y_train_2028)

    # 预测 2028
    pred_log_2028 = final_model.predict(X_test_2028)
    pred_gold_2028 = np.expm1(pred_log_2028)  # 转回实际金牌数

    # 2028 没有真实值 => y_test_2028 全是 NaN => 无法算 RMSE
    results_2028 = test_df_2028[["Year","Sport","Country","GoldMedals"]].copy()
    results_2028.rename(columns={"GoldMedals":"ActualGold"}, inplace=True)
    results_2028["PredictedGold"] = pred_gold_2028

    print("\n=== Predictions for 2028 (head) ===")
    print(results_2028.head(20))

    # 保存到 CSV
    results_2028.to_csv("final_prediction_2028.csv", index=False)
    print("\nPredictions for 2028 saved to 'final_prediction_2028.csv'.")

if __name__ == "__main__":
    main()
