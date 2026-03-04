import pandas as  pd
#返回索引和数据
series = pd.Series([1,2,3,np.nan],index=['a','b','c','d'])
print(series.index)
print(series.values)
#name
series.name = 'test'
print(series.name)
#数据类型
print(series.dtype)
##形状
print(series.shape)
#nan
print(series.hasnans)
#is_unique
print(series.is_unique)
#nbytes
print(series.nbytes)
#axes
print(series.axes)
#ndim
print(series.ndim)
