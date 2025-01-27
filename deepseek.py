import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# 1. 稳健的数据预处理
# ======================
def load_and_filter_data(filepath, predict_year=2028):
    """加载并预处理数据，处理格式异常"""
    try:
        # 加载为原始字符串数据
        raw_data = pd.read_csv(filepath, dtype=str, skipinitialspace=True)
        logger.info(f"成功加载数据，共{len(raw_data)}行")

        # 清洗行尾逗号（保留原始列名）
        clean_data = raw_data.apply(
            lambda col: col.str.replace(r",\s*$", "", regex=True) 
            if col.dtype == object else col
        )

        # 转换数值类型
        clean_data["Year"] = pd.to_numeric(clean_data["Year"], errors="coerce").astype(int)
        country_cols = clean_data.columns[2:]
        clean_data[country_cols] = clean_data[country_cols].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0)

        # 过滤有效数据
        valid_data = clean_data[
            (clean_data["Year"] >= 2000) &  # 保留2000年后的数据
            (clean_data[country_cols].sum(axis=1) > 0)  # 至少有一个国家有奖牌
        ]
        logger.info(f"过滤后数据量：{len(valid_data)}行")

        # 转换为长格式
        long_data = valid_data.melt(
            id_vars=["Year", "Sport"],
            value_vars=country_cols,
            var_name="Country",
            value_name="Gold_Medals"
        )
        long_data["Gold_Medals"] = long_data["Gold_Medals"].astype(int)

        # 添加预测年数据
        new_rows = []
        sports = long_data["Sport"].unique()
        countries = long_data["Country"].unique()
        for sport in sports:
            for country in countries:
                new_rows.append({
                    "Year": predict_year,
                    "Sport": sport,
                    "Country": country,
                    "Gold_Medals": np.nan
                })
        
        return pd.concat([long_data, pd.DataFrame(new_rows)])
    
    except Exception as e:
        logger.error(f"数据加载失败：{str(e)}")
        raise

# ======================
# 2. 稳健的模型构建
# ======================
def build_safe_model(data):
    """构建带异常检查的模型"""
    try:
        # 确保必要列存在
        required_cols = ["Year", "Sport", "Country", "Gold_Medals"]
        assert all(col in data.columns for col in required_cols), "缺失必要列"

        # 创建分类索引
        countries = data["Country"].unique()
        years = data["Year"].unique()
        sports = data["Sport"].unique()
        
        coords = {
            "country": countries,
            "year": years,
            "sport": sports,
            "obs": np.arange(len(data))
        }

        # 索引映射
        data["country_idx"] = data["Country"].map({c:i for i,c in enumerate(countries)})
        data["year_idx"] = data["Year"].map({y:i for i,y in enumerate(years)})
        data["sport_idx"] = data["Sport"].map({s:i for i,s in enumerate(sports)})

        with pm.Model(coords=coords) as model:
            # ===== 数据容器 =====
            country_idx = pm.ConstantData(
                "country_idx", 
                data["country_idx"].values, 
                dims="obs"
            )
            year = pm.ConstantData("year", data["year_idx"].values, dims="obs")
            sport = pm.ConstantData("sport", data["sport_idx"].values, dims="obs")
            
            # ===== 简化模型 =====
            # 国家效应
            country_sd = pm.HalfNormal("country_sd", sigma=1)
            country_effect = pm.Normal("country_effect", 0, country_sd, dims="country")
            
            # 运动基准
            sport_base = pm.Normal("sport_base", 0, 1, dims="sport")
            
            # 线性预测
            log_mu = pm.Deterministic(
                "log_mu",
                country_effect[country_idx] + sport_base[sport],
                dims="obs"
            )
            mu = pm.math.exp(log_mu)
            
            # 似然函数
            pm.Poisson(
                "medals",
                mu=mu,
                observed=data["Gold_Medals"].where(~data["Gold_Medals"].isna()),
                dims="obs"
            )
        
        return model, data
    
    except Exception as e:
        logger.error(f"模型构建失败：{str(e)}")
        raise

# ======================
# 3. 带健康检查的采样
# ======================
def safe_sampling(model, max_retry=3):
    """带错误重试的采样"""
    for attempt in range(max_retry):
        try:
            with model:
                trace = pm.sample(
                    draws=1500,
                    tune=500,
                    chains=2,
                    target_accept=0.9,
                    nuts={"max_treedepth": 12},
                    compute_convergence_checks=False,
                    random_seed=42 + attempt  # 每次尝试不同随机种子
                )
                pm.sample_posterior_predictive(trace, extend_inferencedata=True)
                return trace
        except Exception as e:
            logger.warning(f"采样失败（尝试{attempt+1}/{max_retry}）：{str(e)}")
            if attempt == max_retry - 1:
                raise

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    try:
        # 配置
        DATA_PATH = "/Users/zekaihuang/Desktop/MCM2025/updated_gold_medals_int.csv"
        PREDICT_YEAR = 2028

        # 1. 数据加载
        logger.info("正在加载数据...")
        full_data = load_and_filter_data(DATA_PATH, PREDICT_YEAR)
        
        # 2. 构建模型
        logger.info("构建模型中...")
        model, processed_data = build_safe_model(full_data)
        
        # 3. 采样
        logger.info("开始采样...")
        trace = safe_sampling(model)
        
        # 4. 结果提取
        logger.info("处理结果...")
        is_pred = processed_data["Year"] == PREDICT_YEAR
        pred = az.extract(trace.posterior_predictive["medals"][:, is_pred])
        
        # 按国家汇总
        results = (
            pred.mean("sample")
            .groupby(processed_data.loc[is_pred, "Country"])
            .sum()
            .to_dataframe(name="prediction")
            .sort_values("prediction", ascending=False)
        )
        
        print("\n=== 2028年金牌预测 ===")
        print(results.head(10))
        
    except Exception as e:
        logger.error(f"程序运行失败：{str(e)}")
        print("请检查：")
        print("1. 数据文件路径是否正确")
        print("2. 数据文件格式是否符合要求")
        print("3. 系统内存是否充足（建议>8GB）")