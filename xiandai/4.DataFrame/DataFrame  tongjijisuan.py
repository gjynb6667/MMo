import pandas as pd
import numpy as np
# # 创建一个包含 NaN 值的 DataFrame
# df = pd.DataFrame(
#     {'A': [1, 2, np.nan, 4],
#     'B': [5, np.nan, np.nan, 8],
#     'C': ['foo', 'bar', 'baz', np.nan]
# })
# # 计算每列非 NaN 值的数量
# count_per_column = df.count()
# print("Count per column:")
# print(count_per_column)
# # 计算每行非 NaN 值的数量
# count_per_row = df.count(axis=1)
# print("\nCount per row:")
# print(count_per_row)
# # 只计算数值列的非 NaN 值的数量
# count_numeric_only = df.count(numeric_only=True)
# print("\nCount numeric only:")
# print(count_numeric_only)

# 创建一个包含 NaN 值的 DataFrame
# df = pd.DataFrame({
# 'A': [1, 2, np.nan, 4],
# 'B': [5, np.nan, np.nan, 8],
# 'C': [12, np.nan,np.nan, np.nan]
# })
# # 计算每列的总和
# sum_per_column = df.sum()
# print("Sum per column:")
# print(sum_per_column)
# # 计算每行的总和
# sum_per_row = df.sum(axis='columns')
# print("\nSum per row:")
# print(sum_per_row)
# # 只计算数值列的总和
# sum_numeric_only = df.sum(numeric_only=True)
# print("\nSum numeric only:")
# print(sum_numeric_only)
# # 使用 min_count 参数
# sum_with_min_count = df.sum(min_count=2)
# print("\nSum with min_count=2:")
# print(sum_with_min_count)

# 创建一个包含 NaN 值的 DataFrame
# df = pd.DataFrame({
# 'A': [1, 2, np.nan, 4],
# 'B': [5, np.nan, np.nan, 8],
# 'C': ['foo', 'bar', 'baz', 'qux'] # 非数值列
# })
# # 计算每列的平均值
# # mean_per_column = df.mean()
# # print("Mean per column:")
# # print(mean_per_column)
# # 计算每行的平均值
# # mean_per_row = df.mean(axis='columns')
# # print("\nMean per row:")
#
# # print(mean_per_row)
# # 只计算数值列的平均值
# mean_numeric_only = df.mean(numeric_only=True)
# print("\nMean numeric only:")
# print(mean_numeric_only)

# 创建一个包含 NaN 值的 DataFrame
# df = pd.DataFrame({
# 'A': [1, 2, np.nan, 4],
# 'B': [5, np.nan, 7, 8],
# 'C': [12, 33, 1, 6] # 非数值列
# })
# # 计算每列的中位数
# median_per_column = df.median()
# print("Median per column:")
# print(median_per_column)
# # 计算每行的中位数
# median_per_row = df.median(axis='columns')
# print("\nMedian per row:")
# print(median_per_row)
# # 只计算数值列的中位数
# median_numeric_only = df.median(numeric_only=True)
# print("\nMedian numeric only:")
# print(median_numeric_only)

# 创建一个包含 NaN 值的 DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, np.nan, 4],
#     'B': [5, np.nan, 7, 8],
#     # 'C': ['foo', 'bar', 'baz', 'qux']  # 非数值列
#     'C': [32, 10, 0, 1]  # 非数值列
# })
# # 计算每列的最小值
# min_per_column = df.min()
# print("Min per column:")
# print(min_per_column)
# # 计算每行的最小值
# min_per_row = df.min(axis='columns')
# print("\nMin per row:")
# print(min_per_row)
# # 只计算数值列的最小值
# min_numeric_only = df.min(numeric_only=True)
# print("\nMin numeric only:")
# print(min_numeric_only)

# 创建一个包含 NaN 值的 DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, np.nan, 4],
#     'B': [5, np.nan, 7, 8],
#     'C': ['foo', 'bar', 'baz', 'qux']  # 非数值列
# })
# # 计算每列的最大值
# # max_per_column = df.max()
# # print("Max per column:")
# # print(max_per_column)
# # # 计算每行的最大值
# # max_per_row = df.max(axis='columns')
# # print("\nMax per row:")
# # print(max_per_row)
# # 只计算数值列的最大值
# max_numeric_only = df.max(numeric_only=True)
# print("\nMax numeric only:")
# print(max_numeric_only)

# 创建一个 DataFrame
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [5, np.nan, 3, 2, 6],
    # 'C': ['foo', 'bar', 'baz', 'qux', 'quux']  # 非数值列
})
# 计算每列的累积和
cumsum_per_column = df.cumsum(axis=0)
print("Cumulative sum per column:")
print(cumsum_per_column)
# 计算每行的累积和
cumsum_per_row = df.cumsum(axis=1)
print("\nCumulative sum per row:")
print(cumsum_per_row)