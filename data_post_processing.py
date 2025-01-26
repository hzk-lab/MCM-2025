#!/usr/bin/env python3
import pandas as pd

def main():
    # 1. 读取 2028 年的预测结果 CSV
    #    假设文件名为 final_prediction_2028.csv，且包含:
    #      Year, Sport, Country, ActualGold, PredictedGold
    df = pd.read_csv('/Users/leonhuangzekai/Desktop/MCM/prediction_2024_ignored_real.csv')

    # 2. 按 Country 分组，对 PredictedGold 求和
    df_sum = df.groupby("Country", as_index=False)["PredictedGold"].sum()

    # 3. 你可以把列名改成更明确的名称，如 PredictedGoldTotal
    df_sum.rename(columns={"PredictedGold": "PredictedGoldTotal"}, inplace=True)

    # 4. 可选：对预测值做一下四舍五入，保留小数点后 2~3 位
    # 这里对于金牌我们向下取整
    df_sum["PredictedGoldTotal"] = df_sum["PredictedGoldTotal"].round(3)
    df_sum["PredictedGoldTotalFloor"] = df_sum["PredictedGoldTotal"].apply(lambda x: int(x))

    # 5. 按总金牌数从高到低排序
    df_sum.sort_values("PredictedGoldTotal", ascending=False, inplace=True)

    # 6. 将结果输出到新 CSV
    df_sum.to_csv("prediction_2024.csv", index=False)

    # print("聚合完成: 已输出到 'country_gold_sums_2028_2.0.csv'。")
    print(df_sum.head(20))  # 可选：在控制台上看看前 20 行

if __name__ == "__main__":
    main()
