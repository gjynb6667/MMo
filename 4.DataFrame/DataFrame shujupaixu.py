import numpy as np
import pandas as pd
# 创建一个示例 DataFrame
# df = pd.DataFrame({
#     'col1': ['A', 'A', 'B', np.nan, 'D', 'C'],
#     'col2': [2, 1, 9, 8, 7, 4],
#     'col3': [0, 1, 9, 4, 2, 3],
#     'col4': ['a', 'B', 'c', 'D', 'e', 'F']
# })
# # 打印原始DataFrame
# print(df)
# # 根据 'col1' 列对DataFrame进行排序
# res1 = df.sort_values(by=['col1'])
# # 打印排序后的DataFrame
# print(res1)
# # 根据 'col1' 和 'col2' 列对DataFrame进行排序
# res2 = df.sort_values(by=['col1', 'col2'])
# # 打印排序后的DataFrame
# print(res2)

# 创建一个多级索引的DataFrame
arrays = [np.array(['qux', 'qux', 'foo', 'foo']),
          np.array(['two', 'one', 'two', 'one'])]
df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1]},index=arrays)
print(df)
# 按第一层索引升序排序
df_sorted_by_first_level = df.sort_index(level=0)
print(df_sorted_by_first_level)
# 按第二层索引降序排序
df_sorted_by_second_level_desc = df.sort_index(level=1,ascending=False)
print(df_sorted_by_second_level_desc)
# 按整个索引升序排序
df_sorted_by_full_index = df.sort_index(ascending=True)
print(df_sorted_by_full_index)