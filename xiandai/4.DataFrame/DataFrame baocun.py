#
# import pandas as pd
# # 创建一个简单的DataFrame
# data = {
#     '姓名': ['张三', '李四', '王五'],
#     '年龄': [28, 34, 29],
#     '城市': ['北京', '上海', '广州']
# }
# df = pd.DataFrame(data)
# # 将DataFrame保存为CSV文件
# df.to_csv('人员信息.csv', index=False, encoding='utf_8_sig')

# 创建一个简单的DataFrame
# data = {
#     '姓名': ['张三', '李四', '王五'],
#     '年龄': [28, 34, 29],
#     '城市': ['北京', '上海', '广州']
# }
# df = pd.DataFrame(data)
# # 将DataFrame保存为Excel文件
# df.to_excel('人员信息.xlsx', index=False)

# data = pd.read_csv('./人员信息.csv',nrows=1)
# print(data)


data = pd.read_excel('./人员信息.xlsx')
print(data)

"""
    https://pandas.pydata.org/pandas-docs/
"""