import numpy as np

np.set_printoptions(edgeitems=30, linewidth=10000)
np.set_printoptions(edgeitems=4, linewidth=180)
np.set_printoptions(precision=3)

np.core.arrayprint._line_width = 80


class Matrix:
    """
    Получение обобщенной матрицы жесткостей
    """

    def __init__(self,n_kone):
        """

        :param n_kone: Количество конечных элементов
        :matrix_R: Матрица вставляемая
        :i: Позиция в диагональной матрице целое
        """

        self.__test=self.test_matrix_add(n_kone)
        #self.diagonl_mat=self.mat_umn(matrix_R,i)# Получение диагональной матрицы

    def _get_nul_mat(self):
        return self.__test

    nul_mat=property(fget=_get_nul_mat)
    def test_matrix_add(self, n):
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
        # test = np.diagflat([range(a)])
        #print(test)

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

        print(f"cложен R_22+R_11 \n {self.R_22 + self.R_11}")

        self.k = np.block([[self.R_11, self.R_12], [self.R_21, self.R_22]])
        print(f" Размер массива \n k {self.k}  dim\n{self.k.ndim} \n shape{self.k.shape}")
        return self.k



    def mat_umn(self,matrix_R, i):
        """
        :param test: Матрица нулевая обобщенная  вариант 1 основной
        :param i: Номер конечно элемента
        :return: матрица жесткости
        """""
        m=[]
        m=np.copy(self.nul_mat)
        m[2 * i:(2 * i + 4), 2 * i :2 * i + 4] = matrix_R # 'если вставлять общую матрицу
        #print(f"Матрица жесткости \n {a}")
        #print(a.shape)

        return m

        ''''''''''
    def mat_umn_old(self, test, i):
        """
        
        :param test: Матрица нулевая обобщенная  вариант 2
        :param i: Номер конечно элемента
        :return: матрица жесткости
        """""
        test
        # a = test[4 * i:(4 * i + 4), 4 * i:4 * i + 4]  =self.k# 'если вставлять общую матрицу
        a = test[2 * i:(2 * i + 2), 2 * i:2 * i + 2] = self.R_11
        a1 = test[2 * i + 2:(2 * i + 4), 2 * i:2 * i + 2] = self.R_21
        a2 = test[2 * i:(2 * i + 2), 2 * i + 2:2 * i + 4] = self.R_12
        a3 = test[2 * i + 2:(2 * i + 4), 2 * i + 2:2 * i + 4] = self.R_22  # +R_11(2)
        # Надо заполнить матрицу жесткости а потом складывать матрицы жесткости

        print(f"Матрица жесткости  2\n {test}")
        print(test.shape)
        return test_1
        '''

if __name__ == '__main__':
    n = 10
    mat = Matrix(n)
    mat.mat_Test()
    print(type(mat))

    #test = mat.test_matrix_add(n)  # создание единичной матрицы заполненно 1
    mat.mat_umn( i=2)  # i номер КЭ для которого ищем матрицу до n
   # mat.mat_umn_old(test, i=1)  # номер КЭ для которого ищем матрицу до n
