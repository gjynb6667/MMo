# import pandas as pd
# # 创建一个包含学生信息的列表，每个元素代表一个学生的姓名
# data_list = ['小明', '小红', '小刚']
# # 定义列名
# columns = ['姓名']
# # 使用pandas库创建一个DataFrame，将数据列表和列名作为参数传入
# df = pd.DataFrame(data_list, columns=columns)
# # 打印DataFrame以查看数据
# print(df)
#

# 创建一个包含学生信息的嵌套列表，每个子列表代表一个学生的姓名、年龄和成绩
# data_list = [
#     ['小明', 20, 85],
#     ['小红', 18, 90],
#     ['小刚', 22, 88]
# ]
# # 定义列名，分别对应姓名、年龄和成绩
# columns = ['姓名', '年龄', '成绩']
# # 使用pandas库创建一个DataFrame，将数据列表和列名作为参数传入
# df = pd.DataFrame(data_list, columns=columns)
# # 打印DataFrame以查看数据
# print(df)
# 定义一个字典，其中包含两组数据：姓名和年龄
# data = {
#  'Name': ['Tom', 'Nick', 'John'],# 'Name' 键对应一个包含姓名的列
#  'Age': [20, 21, 19] # 'Age' 键对应一个包含年龄的列
# }
# # 使用pd.DataFrame()函数将字典转换为DataFrame对象
# # 这里，data字典中的键自动成为DataFrame的列名，值成为列的数据
# df = pd.DataFrame(data)
# # 打印DataFrame对象，查看其内容
# print(df)

# 创建三个pandas Series对象
# s1 = pd.Series(['小明', '小红', '小刚'], name='姓名')
# s2 = pd.Series([20, 18, 22], name='年龄')
# s3 = pd.Series([85, 90, 88], name='成绩')
# # 将Series对象组合成一个字典，键是Series的名称，值是Series本身
# # 然后将这个字典传递给DataFrame构造函数来创建一个DataFrame
# df = pd.DataFrame({s1.name: s1, s2.name: s2, s3.name: s3})
# # 打印DataFrame对象，查看其内容
# print(df)
# 创建三个pandas Series对象
s1 = pd.Series(['小明', '小红', '小刚'], name='姓名')
s2 = pd.Series([20, 18, 22, 0], name='年龄')
s3 = pd.Series([85, 90, 88], name='成绩')
# 使用concat拼接，并指定轴为1
df = pd.concat([s1, s2, s3], axis=1)
# 打印DataFrame对象，查看其内容
print(df)