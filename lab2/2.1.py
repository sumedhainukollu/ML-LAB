import numpy as np
list=[[1,0,2],[0,1,1],[2,1,0],[1,1,1],[0,2,1]]
X=np.array(list)
print (X)
mean=np.mean(X)
print(mean)
X_c = X - mean
print("diff between mean and x:")
print(X_c)

# 3. Number of samples
n = X.shape[0]

# 4. Covariance matrix using matrix multiplication
cov_matrix = (X_c.T @ X_c) / (n - 1)
print("Covariance matrix:")
print(cov_matrix)