import numpy as np


n = int(input("Enter number of thetas: "))


thetas = list(map(float, input("Enter thetas : ").split()))


thetas = np.array(thetas)

lamda = 0.01

def l2norm(thetas, lamda):
    return lamda * np.sum(np.square(thetas))

def l1norm(thetas, lamda):
    return lamda * np.sum(np.abs(thetas))


l2 = l2norm(thetas, lamda)
l1 = l1norm(thetas, lamda)

# Print results
print("L2 Regularization:", l2)
print("L1 Regularization:", l1)