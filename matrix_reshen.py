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


class Matrix:
    """
    Получение обобщенной матрицы жесткостей
    """
    def __init__(self):
        test = self.test_matrix_add(14)


    def test_matrix_add(self,n):
        """
        Создание матрицы заполненной 1 по горизонтали
        n: количество кэ
        cоздание единичной матрицы заданной размерности
        :return:
        """

        a = 2 * n + 2
        n -= 0  # порядок матрицы на 1 мень
        a -= 0
        """"""""
        test = np.zeros((a, a))  # Заполнени матрицы нулями
        # test = np.eye(test.shape[0], dtype=int)# Заполнение значениями
        test = np.diagflat([range(a)])
        print(test)
        return test

    def mat_Test(self):
        """
        Функция нужна только для работы с данным моуделем и тестирования
        :return:
        """

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
        self.R_11 = np.block([[a_11, a_12], [a_21, a_22]])
        print(f"R11= \n {self.R_11}")
        print(f" Размер массива {self.R_11.ndim}")
        self.R_22 = np.block([[a_33, a_34], [a_43, a_44]])
        print(f"R_22=\n{self.R_22} {self.R_22.shape}")

        self.R_12 = self.R_21 = np.block([[a_13, a_14], [a_23, a_24]])
        self.R_21 = self.R_21.transpose()
        print(f"R_12= \n{self.R_12}")

        print(f"cложен \n {self.R_22 + self.R_11}")

        self.k = np.block([[self.R_11, self.R_12], [self.R_21, self.R_22]])
        print(f" Размер массива \n k {self.k}  dim\n{self.k.ndim} \n shape{self.k.shape}")
        return self.k


    def mat_umn2(self, test, i=13):
        """
        :param test: 
        :param i: Номер конечно элемента
        :return: матрица жесткости
        """""

        a = test[2 * i:(2 * i + 2), 2 * i:2 * i + 2] = self.R_11
        a1 = test[2 * i + 2:(2 * i + 4), 2 * i:2 * i + 2] = self.R_21
        a2 = test[2 * i:(2 * i + 2), 2 * i + 2:2 * i + 4] = self.R_12
        a3 = test[2*i + 2:(2*i + 4), 2 * i+2 :2 * i + 4] = self.R_22  # +R_11(2)
        # Надо заполнить матрицу жесткости а потом складывать матрицы жесткости

        print(f"test {i},\n {test}")
        print(test)
        print(test.shape)
        return test


if __name__ == '__main__':
    mat = Matrix()
    mat.mat_Test()
    test=mat.test_matrix_add(14)# создание единичной матрицы заполненно 1
    mat.mat_umn2(test,i=12)# i номер КЭ для которого ищем матрицу

