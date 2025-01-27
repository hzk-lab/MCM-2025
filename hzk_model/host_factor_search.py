#!/usr/bin/env python3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import itertools

########################################
# 1. 辅助函数
########################################

def load_host_dict(host_csv):

    df_host = pd.read_csv(host_csv, sep="\t", encoding="utf-8-sig")
    print("Debug: df_host.columns =", df_host.columns)

    host_dict = {}

    for _, row in df_host.iterrows():
        y = row["Year"]
        h = str(row["Host"]).strip()
        # 若 Host 含 "Cancelled" 就跳过
        if "Cancelled" in h:
            continue
        # 否则记录
        host_dict[y] = h
    return host_dict

def adjust_host_medals(df_long, host_dict, alpha):
    """
    对 df_long 进行副本复制，然后把其中 "Year" in host_dict & "Country" == host_dict[Year] 的行
    的 GoldMedals /= alpha.
    返回新的 df_adjusted.
    """
    df_adjusted = df_long.copy()
    mask = df_adjusted.apply(
        lambda row: (row["Year"] in host_dict) and (row["Country"] == host_dict[row["Year"]]),
        axis=1
    )
    df_adjusted.loc[mask, "GoldMedals"] = df_adjusted.loc[mask, "GoldMedals"] / alpha
    return df_adjusted

def apply_host_factor_to_prediction(df_pred, host_dict, alpha):
    """
    对预测结果 df_pred (包含 Year, Country, PredictedGold 等)
    如果 (Year in host_dict) 且 (Country == host_dict[Year])，则 PredictedGold *= alpha.
    用于在最终预测时再乘回去.
    """
    df_final = df_pred.copy()
    mask = df_final.apply(
        lambda row: (row["Year"] in host_dict) and (row["Country"] == host_dict[row["Year"]]),
        axis=1
    )
    df_final.loc[mask, "PredictedGold"] = df_final.loc[mask, "PredictedGold"] * alpha
    return df_final

def calc_rmse_in_original_space(y_true_log, y_pred_log):
    """
    在原始金牌空间里计算RMSE，忽略真实值是NaN的行.
    """
    valid_mask = ~y_true_log.isna()
    if valid_mask.sum() == 0:
        return np.nan
    y_true = np.expm1(y_true_log[valid_mask])
    y_pred = np.expm1(y_pred_log[valid_mask])
    return mean_squared_error(y_true, y_pred, squared=False)

########################################
# 2. 核心逻辑: 滑动窗口 + alpha 搜索
########################################

def main():
    # (a) 读入主办方信息 => 生成 host_dict: Year->HostCountry
    host_dict = load_host_dict("summerOly_hosts.csv")

    # (b) 读入 “宽表” => melt => long
    df_wide = pd.read_csv("updated_gold_medals_int.csv")
    df_long = df_wide.melt(
        id_vars=["Year","Sport"],
        var_name="Country",
        value_name="GoldMedals"
    )

    # (c) 排序 (有些特征工程需先排序)
    df_long = df_long.sort_values(["Country","Sport","Year"]).reset_index(drop=True)

    # (d) 构造一个小函数: 给定 alpha 后，做如下：
    #     1) 调整主办国历史金牌 => gold/alpha
    #     2) 训练 + 验证 (滑动窗口) => 返回总RMSE
    def evaluate_alpha(alpha, target_years, k):
        """
        对给定 alpha, 我们先做 adjust_host_medals( df_long => df_adjusted ),
        再在多窗口 (target_years) 上做滑动窗口+XGBoost, 求平均RMSE.
        """
        df_adj = adjust_host_medals(df_long, host_dict, alpha)  # 历史主办国 = gold/alpha
        
        # 如果有进一步的特征工程(rollingMeanPrev2等), 在 df_adj 上做:
        # 这里只做最简单形式, 如果你原本脚本要 rollingMeanPrev2, 可以在这里加...
        
        # 这里示例: 不做 rollingMean, 只用 (Year, SportEnc, CountryEnc) 来演示
        # ========== LabelEncode
        from sklearn.preprocessing import LabelEncoder
        sport_le = LabelEncoder()
        country_le = LabelEncoder()
        df_adj["SportEnc"] = sport_le.fit_transform(df_adj["Sport"])
        df_adj["CountryEnc"] = country_le.fit_transform(df_adj["Country"])
        # ========== log transform
        df_adj["Target"] = np.log1p(df_adj["GoldMedals"])

        # 定义一个简单的XGBoost超参(或也可再搜)
        params = {"n_estimators":100, "max_depth":4, "learning_rate":0.1, "random_state":42}

        def make_sliding_window_data(df, tyear, k_):
            all_years = sorted(df["Year"].unique())
            if tyear not in all_years:
                return None,None
            idx = all_years.index(tyear)
            start_idx = max(0, idx - k_)
            train_years = all_years[start_idx:idx]
            train_df = df[df["Year"].isin(train_years)].copy()
            test_df  = df[df["Year"] == tyear].copy()
            return train_df, test_df

        def train_and_eval_one_window(train_df, test_df):
            # 构造特征 & target
            feat_cols = ["Year","SportEnc","CountryEnc"]  # 演示: 只用这几个
            X_train = train_df[feat_cols]
            y_train = train_df["Target"]
            X_test  = test_df[feat_cols]
            y_test  = test_df["Target"]

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            y_pred_log = model.predict(X_test)
            rmse = calc_rmse_in_original_space(y_test, pd.Series(y_pred_log))
            return rmse
        
        rmse_list = []
        for yy in target_years:
            trdf, tsdf = make_sliding_window_data(df_adj, yy, k)
            if trdf is None or tsdf is None:
                continue
            rmse_yy = train_and_eval_one_window(trdf, tsdf)
            if not np.isnan(rmse_yy):
                rmse_list.append(rmse_yy)

        if len(rmse_list)==0:
            return np.nan
        return np.mean(rmse_list)

    # (e) 搜索 alpha
    alpha_candidates = np.arange(1.0, 1.51, 0.1)  # 1.0 ~ 1.5, 步长 0.1
    target_years_for_eval = [2012, 2016, 2020, 2024]
    k_window = 3

    best_alpha = None
    best_rmse = float("inf")

    for alpha in alpha_candidates:
        rmse_val = evaluate_alpha(alpha, target_years_for_eval, k_window)
        print(f"alpha={alpha}, meanRMSE={rmse_val}")
        if not np.isnan(rmse_val) and rmse_val < best_rmse:
            best_rmse = rmse_val
            best_alpha = alpha

    print(f"\nBest alpha={best_alpha}, with RMSE={best_rmse}")

    # (f) 用 best_alpha 再做一次全训练, 并预测 2028
    # 实际流程里，你可能先把 df 调整 => train, model => predict => df_pred,
    # 然后再 apply_host_factor_to_prediction(df_pred, host_dict, best_alpha)

    # 这里只给个概念：比如你最终对 2028 做预测后, results_2028 = ...
    # results_2028 = apply_host_factor_to_prediction(results_2028, host_dict, best_alpha)
    # results_2028.to_csv("final_pred_2028_with_host.csv", index=False)

    print("Done.")

if __name__ == "__main__":
    main()
