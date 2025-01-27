import pandas as pd

# 读取三个 CSV 文件
df1 = pd.read_csv('updated_gold_medals.csv')
df2 = pd.read_csv('updated_silver_medals.csv')
df3 = pd.read_csv('updated_bronze_medals.csv')

# 以第三个文件的列和索引为标准
standard_columns = df3.columns
standard_index = df3.index

# 确保所有文件的列名和索引对齐，缺失部分填充为 0
df1 = df1.reindex(index=standard_index, columns=standard_columns, fill_value=0)
df2 = df2.reindex(index=standard_index, columns=standard_columns, fill_value=0)

# 提取前两列和第一行
static_columns = df3.iloc[:, :2]  # 前两列（年份和项目）
static_row = df3.iloc[0, :]      # 第一行（国家名称）

# 对数据部分逐项相加
summed_values = df1.iloc[1:, 2:].astype(float) + df2.iloc[1:, 2:].astype(float) + df3.iloc[1:, 2:].astype(float)

# 拼接最终结果
result = pd.concat([static_columns, summed_values], axis=1)
result.columns = standard_columns  # 恢复原列名
result.iloc[0, :] = static_row     # 恢复第一行（国家名称）


# 保存结果到 CSV
output_file = 'updated_total_medals.csv'  # 替换为你的输出文件路径
result.to_csv(output_file, index=False)
