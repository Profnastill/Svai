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
    R_11 = np.block([[a_11, a_12], [a_21, a_22]])
    print(f"R11 \n {R_11}")

    R_22 = np.block([[a_33, a_34], [a_43, a_44]])
    R_12 = R_21 = np.block([[a_13, a_14], [a_23, a_24]])
    print(f" Размер массива {R_11.ndim}")

    k = np.block([[R_11, R_12], [R_21, R_22]])
    print(f" Размер массива \n k {k}  dim\n{k.ndim} \n shape{k.shape}")
    return k


def mat_umn(test, m):
    for i in m:
        print(m)
        test = np.eye(test.shape[0], dtype=int) * m
        print(f" test {i},{test}")
    return test


def mat_umn2(test, m):
    print(f" m {m.shape}")
    n = int(test.shape[0] / m.shape[0])
    print(m)

    for i in range(n):
        print(i)

    # test[i * 4:(i + 1) * 4, i * 4:(i + 1) * 4] += m

    test2= test[(i+3) * 2: (i+1+3) * 2, (i+3) * 2:(i + 1) * 2]
    print(test2)

    test[i:(i+1)*2:-2*(i+1)]=m[0:2,:-2]# R11


    test[i * 4: (i + 1) * 4, i * 4:(i + 1) * 4] = +m



    # test[i * 4: (i + 1) * 4, i * 4:(i + 1) * 4] = +m
    test.shape
    print(f"test {i},\n {test}")
    print(test.shape)

    return test


if __name__ == '__main__':
    k = mat()

    diagonal_1 = np.diagonal(k).tolist()
    print(f"Диагональ \n{k}")

    n = 13  # +4 конечных
    a = 2 * n + 2
    n -= 0  # порядок матрицы на 1 мень
    a -= 0
    """"""""
    test = np.zeros((a, a))
    print(test)

    test = np.eye(test.shape[0], dtype=int)

    print(test)
    print("-------", type(diagonal_1))



    a = mat_umn2(test, k)


















def ger(i, n):
    i += 1
    if i <= n:
        i = ger(i, n)
    else:
        i

    return i


a = ger(6, 20)
print(a)


def dsgfs():
    b = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    a = np.zeros((9, 9), int)
    np.fill_diagonal(a, b)
    np.eye(a.shape[0], dtype=int) * b
