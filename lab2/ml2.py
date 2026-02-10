import numpy as np


X1=np.linspace(-10,10,1)
X2=np.linspace(10,30,1)
X3=np.linspace(30,50,1)

gradient=[]

y=(2*X1)+(3*X2)+(3*X3)+4

dy_x1=2*1*(X1**0)
list1=[dy_x1]
gradient.append(list1)
dy_x2=3*1*(X2**0)
list2=[dy_x2]
gradient.append(list2)
dy_x3=3*1*(X3**0)
list3=[dy_x3]
gradient.append(list3)

print(gradient)
