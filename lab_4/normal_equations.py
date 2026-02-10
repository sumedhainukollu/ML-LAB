import numpy as np
x=np.array([[1,2],
           [2,3],
           [3,4]])
y=np.array([[5],
           [6],
           [7]])
#design matrix
xbar=x.transpose()


print(xbar)


#cost function
xr=xbar @ x
x_inv = np.linalg.inv(xr)

theta=x_inv @ xbar @ y
print(theta)
