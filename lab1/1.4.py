import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

mean=0
sigma=15

X=np.linspace(-100,100,100)

Y=norm.pdf(X,loc=mean,scale=sigma)

plt.plot(X,Y)
plt.title("gausian distribution")
plt.show()