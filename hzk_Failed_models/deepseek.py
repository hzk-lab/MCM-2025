import pandas as pd
import pymc as pm
import numpy as np

# 1. Load data
data = pd.read_csv('/Users/zekaihuang/Desktop/MCM2025/updated_gold_medals_int.csv', skipinitialspace=True)

# 转换为长格式
country_codes = data.columns[2:].tolist()  # 获取国家三字码列表
long_data = data.melt(
    id_vars=["Year", "Sport"],
    value_vars=country_codes,
    var_name="Country",
    value_name="Gold_Medals"
)

# 转换数据类型
long_data["Gold_Medals"] = (
    long_data["Gold_Medals"]
    .astype(str).str.strip()
    .replace({"": "0"})
    .astype(int)
)

# --------------------------
# 2. 创建分类索引（关键修正！）
# --------------------------
# 将分类变量转换为索引
categories = {
    "country": sorted(long_data["Country"].unique()),
    "year": sorted(long_data["Year"].unique()),
    "sport": sorted(long_data["Sport"].unique())
}

# 添加索引列
long_data["country_idx"] = long_data["Country"].map(
    {name: idx for idx, name in enumerate(categories["country"])}
)
long_data["year_idx"] = long_data["Year"].map(
    {year: idx for idx, year in enumerate(categories["year"])}
)
long_data["sport_idx"] = long_data["Sport"].map(
    {sport: idx for idx, sport in enumerate(categories["sport"])}
)

# --------------------------
# 3. 添加2028年预测数据
# --------------------------
new_year = 2028
new_rows = []
for sport in categories["sport"]:
    for country in categories["country"]:
        new_rows.append({
            "Year": new_year,
            "Sport": sport,
            "Country": country,
            "Gold_Medals": np.nan,
            "country_idx": categories["country"].index(country),
            "year_idx": len(categories["year"]),  # 2028年是新索引
            "sport_idx": categories["sport"].index(sport)
        })

# 更新年份分类
categories["year"].append(new_year)
full_data = pd.concat([long_data, pd.DataFrame(new_rows)])

# --------------------------
# 4. 定义坐标系统
# --------------------------
coords = {
    "country": categories["country"],
    "year": categories["year"],
    "sport": categories["sport"],
    "obs": np.arange(len(full_data))
}

# --------------------------
# 5. 构建模型（使用数值索引）
# --------------------------
with pm.Model(coords=coords) as model:
    # 数据容器（使用数值索引）
    country_idx = pm.Data("country_idx", full_data["country_idx"].values, dims="obs")
    year_idx = pm.Data("year_idx", full_data["year_idx"].values, dims="obs")
    sport_idx = pm.Data("sport_idx", full_data["sport_idx"].values, dims="obs")
    
    # 分层效应
    global_mean = pm.Normal("global_mean", mu=0, sigma=2)
    
    # 国家效应
    country_sd = pm.HalfNormal("country_sd", sigma=1)
    country_effect_raw = pm.Normal("country_effect_raw", 0, 1, dims="country")
    country_effect = pm.Deterministic("country_effect", country_effect_raw * country_sd)
    
    # 年份效应
    year_sd = pm.HalfNormal("year_sd", sigma=1)
    year_effect_raw = pm.Normal("year_effect_raw", 0, 1, dims="year")
    year_effect = pm.Deterministic("year_effect", year_effect_raw * year_sd)
    
    # 运动效应
    sport_sd = pm.HalfNormal("sport_sd", sigma=1)
    sport_effect_raw = pm.Normal("sport_effect_raw", 0, 1, dims="sport")
    sport_effect = pm.Deterministic("sport_effect", sport_effect_raw * sport_sd)
    
    # 线性预测器
    log_mu = (
        global_mean
        + country_effect[country_idx]
        + year_effect[year_idx]
        + sport_effect[sport_idx]
    )
    mu = pm.math.exp(log_mu)
    
    # 负二项似然
    alpha = pm.Exponential("alpha", 1)
    pm.NegativeBinomial(
        "medals",
        mu=mu,
        alpha=alpha,
        observed=full_data["Gold_Medals"].where(~full_data["Gold_Medals"].isna()),
        dims="obs"
    )
    
    trace = pm.sample(2000, tune=1000, target_accept=0.95)

# --------------------------
# 6. 预测结果提取
# --------------------------
# 获取2028年数据索引
is_2028 = full_data["Year"] == new_year

# 后验预测
with model:
    pm.sample_posterior_predictive(trace, extend_inferencedata=True)

# 按国家汇总
pred_medals = (
    az.extract(trace.posterior_predictive)
    ["medals"][:, is_2028]
    .mean(axis=0)
    .groupby(full_data.loc[is_2028, "Country"])
    .sum()
    .sort_values(ascending=False)
)

print("2028年金牌预测（前10名）：\n", pred_medals.head(10))