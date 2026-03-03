import pandas as pd
# # 创建一个 DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, 3, 4, 5],
#     'B': ['a', 'b', 'a', 'b', 'a']
# })
# # 打印原始DataFrame
# print(df)
# '''
#     单一值替换
# '''
# # 用数字 100 替换所有的 1
# df_replaced = df.replace(to_replace=1, value=100)
# print('单一值替换\n',df_replaced)
# '''
#     列表替换所有匹配值
# '''
# df_replaced = df.replace(to_replace=[2,3,'a'], value='z')
# print('列表替换所有匹配值\n',df_replaced)
# '''
#     字典替换所有匹配值
# '''
# df_replaced = ({
#      2: 200,
#     'b': 'y'
# })
# df_replaced = df.replace(to_replace=df_replaced)
# print('字典替换所有匹配值\n',df_replaced)
#
# # 使用正则表达式替换
# import pandas as pd
#
# df = pd.DataFrame({
#     'col1': ['apple', 'banana', 'cherry', 'agerape', 'apricote'],
#     'col2': ['apple pie', 'banana split', 'cherry tart', 'grape juice', 'apricote jam']
# })
# """
#     ^：匹配字符串的开始。
#     a：匹配字符 "a"。
#     .*：匹配任意数量的字符（包括零个字符）。
#     e：匹配字符 "e"。
#     $：匹配字符串的结束。
# """
# df_replaced = df.replace(to_replace=r'^a.*e$', value='fruit', regex=True)
# print(df_replaced)


# 创建一个示例 DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4.5, 5.5, 6.5],
    'C': ['7', '8', '9']
})
# 打印原始DataFrame
print(df)
# 将列 'A' 转换为浮点数类型
c = df['A'].astype(float)
print(c)
# 使用字典将多列转换为不同的数据类型
# 将列 'B' 转换为整数类型，列 'C' 也转换为整数类型
c = df.astype({
    'B': int,
    'C': int
})

# 打印DataFrame中各列的数据类型
print(c)

c = df.astype('float64')
print(c)