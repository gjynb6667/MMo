import  numpy as  np
X = np.array([[0,2],[1,1],[2,0]])
X = X.T
print(np.cov(X))
mean_x1 = 1
mean_x2 = 1
cov_x1x1 = np.sum((X[0,:]-mean_x1)**2)/(X.shape[1]-1)
cov_x1x2 = np.sum((X[0,:]-mean_x1)*(X[1,:]-mean_x2))/(X.shape[1]-1)
cov_x2x1 = cov_x1x2
cov_x2x2 = np.sum((X[1,:]-mean_x2)**2)/(X.shape[1]-1)
