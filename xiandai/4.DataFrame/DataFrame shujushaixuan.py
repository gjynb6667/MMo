import pandas as pd
data = {
    '姓名': ['小明', '小红', '小刚'],
    '年龄': [20, 18, 22],
    '成绩': [85, 90, 88]
}
df = pd.DataFrame(data)
print(df['成绩'] >= 90)
# 使用布尔索引选择成绩大于或等于90的学生
high_scores = df[df['成绩'] >= 90]
print(high_scores) 