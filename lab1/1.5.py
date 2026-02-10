import numpy as np
import matplotlib.pyplot as plt

X=np.linspace(-10,10,100)
y1=X*X
y2=2*X

x1=[-5,-3,0,3,5]
Y=[]
for i in x1:
    y=2*i
    Y.append(y)

plt.scatter(x1, Y, c='blue', label='All Points')
print(Y)



plt.plot(X,y1)
plt.plot(X,y2)
plt.show()