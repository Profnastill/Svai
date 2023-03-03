import numpy as np

a=np.array([[1,2],[3,4]])
b=np.array([44,4,4,3,4,5,])
print(a,b)
U=np.dot(b * a)


print(U)
U = np.dot(np.linalg.matrix_power(a, -1) * b)
print(U)