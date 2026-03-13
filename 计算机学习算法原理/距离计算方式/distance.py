import  math
#欧式距离
def eucalidean_siatance(x,y):
    return math.sqrt(sum([(a-b)**2 for a,b in zip(x,y)]))
#曼哈顿距离
def manhattan_distance(x,y):
    return sum([abs(a-b) for a,b in zip(x,y)])
x = [1,2]
y = [4,6]
print(eucalidean_siatance(x,y))
print(manhattan_distance(x,y))
