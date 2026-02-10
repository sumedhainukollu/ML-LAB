import numpy as np
X=np.array([[1,2],
           [3,4],
           [4,5]])
y=np.array([[5],
             [6],
             [7]
            ])
print(X)
print(y)

print(len(X))

#hypothesis function
theta_0=0
theta_1=0

#x_bar=np.array(X)
#y_bar=np.array(y)


for j in range(100):
    y_pred=[]
    print("after" ,j ,"iterations" )
    for i in range(len(X)):
        hypothesis = (theta_0*X[i][0]) + (theta_1*X[i][1])
        y_pred.append(hypothesis)


    costfun=0
    for i in range ((len(X))):
        cost_fun=costfun+((y_pred[i]-y[i]*y_pred[i]-y[i]))

#gradient descent
    for k in range ((len(X))):
        lr=0.00001
        gd_0=((y_pred)-(y)*X[i][0])
        gd_1=((y_pred)-(y)*X[i][1])
        #gd_2=((y_pred)-(y)*X[i][2])
        theta_0=theta_0*lr * gd_0
        theta_1 = theta_1 - lr * gd_1

        print(theta_0)
        print(theta_1)