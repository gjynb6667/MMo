import pandas as pd
import numpy as np
# # 创建一个包含缺失值的 DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, np.nan],
#     'B': [4, np.nan, 6],
#     'C': [7, 8, 9]
# })
# # 打印原始DataFrame
# print(df)
# # 使用 isnull() 方法检测缺失值
# missing_values = df.isnull()
# print(missing_values)
# import pandas as pd
# import numpy as np
# 创建一个包含缺失值的 DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, np.nan],
#     'B': [4, np.nan, 6],
#     'C': [7, 8, 9]
# })
# # 打印原始DataFrame
# print(df)
# # 删除任何含有 NaN 值的行
# df_cleaned = df.dropna()
# print(df_cleaned)

# 创建一个包含缺失值的 DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, np.nan],
#     'B': [np.nan, np.nan, 6],
#     'C': [7, np.nan, 9]
# })
# '''
#     标量填充
# '''
# # 打印原始DataFrame
# print(df)
# # 使用固定值填充缺失值
# df_filled_value = df.fillna(value=0)
# print(df_filled_value)
#
#
# '''
#     前向填充
# '''
# df_filled_value = df.fillna(method='ffill')
# print(df_filled_value)
#
# '''
#     后向填充
# '''
# df_filled_value = df.fillna(method='bfill')
# print(df_filled_value)
#
# """
#     指定列标签填充
# """
# data = {
#     'A': 'a',
#     'B': 'b',
#     'C': 'c'
# }
# df_filled_value = df.fillna(value=data)
# print(df_filled_value)
#
# """
#     使用 limit 参数
# """
#
# filled_with_limit = df.fillna(value=0, limit=1)
# print(filled_with_limit)

# 创建一个包含重复行的 DataFrame
df = pd.DataFrame({
    'A': [1, 1, 2, 2, 3, 3],
    'B': [1, 1, 2, 2, 3, 3],
    'C': [1, 1, 2, 2, 3, 3]
})
# 打印原始DataFrame
print(df)
# 删除重复行，保留第一次出现的重复项
df_dedup_first = df.drop_duplicates()
print(df_dedup_first)
# 根据指定列删除重复行
df_dedup_column = df.drop_duplicates(subset=['A'])
print(df_dedup_column)