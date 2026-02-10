#sigmoid_function
import numpy as np
import matplotlib.pyplot as plt
# Source - https://stackoverflow.com/q/55959156

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

#values of x
x=np.linspace(-10,10,100)
y=Sigmoid(x)

plt.plot(x, y, label='$\sigmoid_func', color='blue', linewidth=2)



plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('\sigma(x)$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# Save and display
plt.savefig('sigmoid_plot.png')


