import  numpy as  np
import  matplotlib.pyplot as plt
import  pandas as  ps
#先读取文档
df = ps.read_excel('./source.xlsx')

#把文档中空白的数据用0填充
df = df.fillna(0)
#获取exam的数据
exam_data = df['exam'].values
#获取attendance的数据
attendance_data = df['attendance'].values
#通过round函数四舍五入去掉小数
finallu_data = np.round(exam_data*0.7+attendance_data*0.3)
#添加finallul列
df['finallu'] = finallu_data
#通过lambda来判断列表里的数是否大于60
df['pass'] = df['finallu'].apply(lambda x:'yes' if x>= 60 else 'no')
#存放到一个新表里
# df.to_excel('./source1.xlsx')
bins = np.arange(0,111,10)
#通过histogram函数可以得到，在不同范围的数量和范围
hist,bin_edges = np.histogram(df['finallu'],bins = bins)
print(bin_edges)
fig = plt.figure()
#得到宽度
bar_width = (bin_edges[1] - bin_edges[0])
plt.bar(bin_edges[:-1],hist,width = bar_width,align = 'edge')
for i in range(len(hist)):
    if hist[i]:
        plt.text(bin_edges[i]+bar_width/2,hist[i]+0.1,str(hist[i]),ha = 'center')
plt.title('finallu')
plt.xlabel('score')
plt.ylabel('number')
plt.xticks(bin_edges[:-1])
plt.show()
#创建一个新的图标实例
fig = plt.figure()
#通过value_counts函数得到yes和no的数量
pass_count = df['pass'].value_counts()
plt.pie(pass_count,labels = pass_count.index,autopct = '%1.1f')
plt.title('Distribution of Passing Status')
plt.show()