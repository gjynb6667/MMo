import pandas as pd
# # 创建一个Series
# series = pd.Series([10, 20, 30, 40, 50])
# # 通过位置索引获取元素
# print(series)
# print(series[0])
# print(series[2])
# 创建一个带有标签的Series
# series = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c','d', 0])
# # 通过标签索引获取元素
# print(series['a'])
# print(series['c'])
# print(series[0])   # 之所以能通过下标访问，是因为之前的标签就是从0开始，一次递增

# 创建一个带有标签的Series
series = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c','d', 'e'])
# 通过位置切片
print(series[:])
# 通过标签切片
print(series['b':'d'])