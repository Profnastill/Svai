import numpy as np

np.set_printoptions(edgeitems=30, linewidth=10000)
np.set_printoptions(edgeitems=4, linewidth=180)
np.set_printoptions(precision=3)

np.core.arrayprint._line_width = 80
a = np.array([[1, 2], [3, 4]])
print(a.shape)
b = np.array([44, 4, 4, 3, 4, 5, ])

b = np.diagflat([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
b = np.diagflat([(5, 4, 6), (6, 5, 26), (7, 8, 9)])


def mat():
    a_11 = 1
    a_12 = a_21 = 2
    a_22 = 3
    a_33 = 4
    a_34 = a_43 = 5
    a_44 = 6
    a_13 = a_31 = 7
    a_14 = a_41 = 8
    a_23 = a_32 = 9
    a_24 = a_42 = 10
    R_11 = np.array([[a_11, a_12], [a_21, a_22]])
    R_22 = np.array([[a_33, a_34], [a_43, a_44]])
    R_12 = R_21 = np.array([[a_13, a_14], [a_23, a_24]])

    k = np.array([[R_11, R_12], [R_21, R_22]])

    print(k)

    return k


def mat_umn(test,m):
    for i in m:
        print(m)
        test = np.eye(test.shape[0], dtype=int) * m
    return  test

def mat_umn2(test,m):
    print(test[0:2,0:2],"\n----")

    print(m[0:1])
    print(m.shape)

    test[0:2,0:2]=np.array([[1,2],[3,4]])
    print("----")
    print(test)

    return  test



if __name__ == '__main__':
    k=mat()

    diagonal_1 = np.diagonal(k).tolist()
    print(f"Диагональ \n{k}")

    n = 4
    a = 2 * n + 2
    n -= 0  # порядок матрицы на 1 мень
    a -= 0
    """"""""
    test = np.zeros((n, a))

    test=np.eye(test.shape[0],dtype=int)
    print(type(diagonal_1))
    a=mat_umn2(test , k)



def dsgfs():
    b = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    a = np.zeros((9, 9), int)
    np.fill_diagonal(a, b)
    np.eye(a.shape[0], dtype=int) * b
