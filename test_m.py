import numpy as np

np.set_printoptions(edgeitems=30, linewidth=10000)
np.set_printoptions(edgeitems=4, linewidth=180)
np.set_printoptions(precision=3)

np.core.arrayprint._line_width = 80

b = np.diagflat([(1, 2, 3, 4)])
matr = np.eye(4, 4)
print(matr)
print("----")
print(b)
print("----")
print(b[2:4:])
print("----")
print(b[2:4,2:])# запятая говорит что по вертикали R22
print("----")
print(b[0:2,:-2])#  Знак минус разворачивает направление среза R11
print("----")
print(b[2:4,:-2])# R21
print("----")

print(b[0:2,2:])# R12

