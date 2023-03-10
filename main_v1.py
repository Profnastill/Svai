##
import math

import numpy
import numpy as np
import pandas as pd
import xlwings as xw

import matrix_reshen
import matrix_reshen as mr

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 60
pd.options.display.expand_frame_repr = False
np.set_printoptions(edgeitems=30, linewidth=10000)
np.set_printoptions(threshold=100000)
np.set_printoptions(precision=3)

np.core.arrayprint._line_width = 4555
# movies.head()


beton_type = ["В3,5", "В5", "В7,5", "В10", "В12,5", "В15", "В20", "В25", "В30", "В35", "В40", "В45", "В50", "В55",
              "В60", "В70", "B80", "B90", "В100"]

beton_Rbn = [2.7, 3.5, 5.5, 7.5, 9.5, 11, 15, 18.5, 22, 25.5, 29, 32, 36, 39.5, 43, 50, 57, 64, 71]

data_Rb = pd.DataFrame(index=beton_type, data=beton_Rbn, columns=["Rb"])

print(data_Rb["Rb"]["В30"])
print(data_Rb)

lc_l = [4, 5, 6, 7, 8, 9, 10]
w = [0.77, 0.68, 0.61, 0.56, 0.53, 0.51, 0.49]

data_w = pd.DataFrame(index=lc_l, data=w, columns=["lc_bi"])

print(data_w)

print(data_w["lc_bi"][10])


class SvAi:
    def __init__(self, type_sv, P, M, N, l, b1, h1, b2, h2, Class_Bet, Class_arm, As, As_, As2, As2_, Zondir):
        """
        __int__(self, type_sv, P, M, N, l, b1, h1, b2, h2, Class_Bet, As, As_):
        :param type_sv:
        :param P: Горизонтальная сила Н
        :param M: момент Н*мм
        :param N: Продольная сила Н
        :param l: Длина сваи
        :param n: Количество конечных элементов
        :param b1,b2: Ширина верха и низа сваи соответственно
        :param h1,h2: Высота верха и низа сваи соответственно
        :return:
        """
        print("табл")
        self.ln_elem = 200  # Длина конечного элемента мм
        self.type_sv = type_sv
        self.P = P * 1000 # Перевод из кН в н
        self.M = M * 1000 * 100  # Н*мм
        self.N = N * 1000  # Продольная сила Н
        self.l = l * 1000  # длина сваи мм
        self.b1 = b1  # мм
        self.b2 = b1  # мм
        self.h1 = h1  # мм
        self.h2 = h1  # мм
        gamma_b = 1.3
        self.k_zondir = (1 if Zondir == "Стабилизация" else 0.6)  # Коэффициент зондирования
        self.Rb = float(data_Rb["Rb"][Class_Bet]) * gamma_b
        self.Eb = 3 * 10 ** 4  # модуль упругости бетона Н/мм2 МПа!!!!!!
        self.Ea = 206000  # Н/мм2 модуль арматуры не напрягаемой
        self.As = As * 100  # лощадь арматуры мм2
        self.As_ = As_ * 100  # площадь растянутой арматуры мм2
        self.As2 = As2 * 100  # лощадь арматуры мм2
        self.As2_ = As2_ * 100  # площадь растянутой арматуры мм2
        self.a = 30  # Толщина защитного слоя    мм
        self.a_n = 40  # толщина защитного слоя для напрягаемой арматуры мм

        self.table_bs = self.fun_Setka(data_grunt).copy()  # Разбиваем грунт на конечные элементы

        self.Ar_sum = self.table_bs["Ar"].sum()


        self.fun_Bi()  # Жесткость элемента

        print(self.table_bs)

        self.len_matr = len(self.table_bs)  # Количество конечных элементов

        self.table_bs["K"] = self.table_bs.apply(lambda row: self.Koeffic_Postely(row),
                                                 axis=1)  # определяем коэффициент постели
        self.table_bs["B"] = self.table_bs.apply(lambda row: self.matrix_B_piramida(row), axis=1)  # Матрица жесткости
        print(self.table_bs.index)
        K_ob = self.MaTrix()  # Получение диагональной  обобщенной матрицы жесткости

        matrix_force = self.fun_Matrix_Force()
        u = self.fun_Matrix_u(matrix_force, K_ob)
        print(f"Перемещения в узлах \n {u}")


    def MaTrix(self):
        """
        Получение обобщенной диагональной  матрицы жестскости
        :return:numpy array  Обобщенная матрица
        """
        print("-------------")
        self.table_bs["K_ob"] = 0
        mr = matrix_reshen.Matrix(self.len_matr)  # Запускаем создания матрицы
        K_ob = mr.test  # Получаем нулевую матрицу
        for i in self.table_bs.index:
            K_ob += mr.mat_umn(self.table_bs.iloc[i]["B"], i)  # получение обобщенной матрицы для каждой строки
        print(f"----Матрица --- \n   {K_ob} ")

        return K_ob

    def fun_Bi(self):
        """
        Определение коэффициентов
        :return:
        """

        self.table_bs["x_2"] =self.table_bs["x"].sum()
        self.table_bs["x_1"] = self.table_bs["x"].shift(1,fill_value=0).sum()

        self.table_bs["fi1"] = self.b1 - (self.b1 - self.b2) / self.l * (
                self.table_bs["x_1"] + self.table_bs["x_2"])

        self.table_bs["fi2"] = self.b1 - (self.b1 - self.b2) / self.l * (
                self.table_bs["x_1"] - self.table_bs["x_2"])

        self.table_bs["bi"] = 2*self.table_bs["fi1"]

        self.table_bs['Bi__'] = (35 * self.table_bs["fi1"] ** 4) + (
                126 * self.table_bs["fi1"] ** 2) + (self.table_bs["fi2"] ** 2) + (
                                        15 * self.table_bs["fi2"] ** 4)
        self.table_bs['Bi__1'] = (35 * self.table_bs["fi1"] ** 4) + (
                154 * self.table_bs["fi1"] ** 2 + self.table_bs["fi2"] ** 2) + (
                                         19 * self.table_bs["fi2"] ** 4)
        self.table_bs['Bi__2'] = (35 * self.table_bs["fi1"] ** 4) + (
                112 * self.table_bs["fi1"] ** 2 + self.table_bs["fi2"] ** 2) + (
                                         13 * self.table_bs["fi2"] ** 4)
        self.table_bs['Bi__3'] = (70 * self.table_bs["fi1"] ** 4) + (
                42 * self.table_bs["fi1"] ** 2 + self.table_bs["fi2"] ** 3)
        self.table_bs

    def fun_Setka(self, data_grunt):
        """
        Разбивка сетки конечных элементов
        :return:
        """
        data_grunt: pd.DataFrame
        print(data_grunt)
        lis = []
        print("dfs")
        da=pd.DataFrame(columns=data_grunt.columns)
        last=0
        for i in range(len(data_grunt)):
            self.ln_elem=10
            col = np.linspace(0, data_grunt.iloc[i]["lsv"], self.ln_elem)

            col=col.tolist()
            col=list(map(lambda x:x+last,col))

            f= [data_grunt.iloc[i]] * (len(col)-1)  # Размножаем строку таблицы грунта
            da=da.append(f,ignore_index=True)
            lis.extend(col)# Список координат свай

            last = lis[-1]
            lis.pop()

        da["x"]=lis#  Координаты начала и конца конечного элемента
        da["lкэ"]=da["x"].diff()

        return da

    def Koeffic_Postely(self, table: pd.DataFrame):
        """
        :param table:
        :return: коэффициент постели
        """
        print(table)
        print("----")
        print(f"l={self.l},{table['bi']}")
        self.w = self.l // table["bi"]
        print("-----------w= ", self.w)
        if self.w < 10:
            self.w = data_w.query("lc_bi==@self.w")
        elif self.w >= 10:
            self.w = data_w["lc_bi"][10]
        print("w=", self.w)

        if table["Er"] != None and table["Er"] != 0:
            k = table["Ar"] * table["Er"] * self.w / (1 - table["Nu"] ** 2) * table["bi"]
        else:
            k = self.k_zondir * self.Ar_sum * self.w / ((1 - table["Nu"] ** 2)) * table["bi"] * (
                    5.5 * table["qi"] + (table["qi"] / 50) ** 2)

        return k

    def Gest_Sechen(self, table: pd.DataFrame):
        """
        Жесткость элемента
        :return:
        """
        na = self.Ea / self.Eb
        nn = self.Ea / self.Eb
        B = table["b"] * table["h"] ** 3 / 12 + (na * self.As * (table["h"] - self.a) ** 2) + nn * self.As2 * (
                table["h"] - self.a_n) ** 2 * 0.85 * self.Eb
        return B

    def fun_Matrix_Force(self):
        matrix_force = np.array([[self.P], [self.M]])
        matrix_force = np.append(matrix_force, np.zeros((self.len_matr * 2, 1)))
        print("-----------")
        print(type(matrix_force))
        return matrix_force

    def matrix_B_piramida(self, table: pd.DataFrame):
        """
        Матрица жесткостей
        :param table:
        :return: Матрица жесткости
        """
        # print(table)
        # I = self.Moment_inerc(table["b"], table["h"])
        self.dzeta = 1
        a_11 = 1 / 560 * (self.dzeta * ((self.Eb / table["x"] ** 3) * table["Bi__"]) + 8 *
                          table["x"] * table["K"] * (
                                  13 * table["fi1"] + 7 * table["fi2"]))

        a_12 = a_21 = 1 / 3360 * (self.dzeta * ((self.Eb / table["x"] ** 2) * (
                3 * table["Bi__"] + 2 * table["Bi__3"])) + 8 * (table[
                                                                    "x"] ** 2) *
                                  table["K"] * (
                                          11 * table["fi1"] + 4 * table["fi2"]))
        a_13 = a_31 = 1 / 560 * (
                -1 * self.dzeta * (self.Eb / table["x"] ** 3) * table["Bi__"] + 36 *
                table["x"] * table["K"] * table["fi1"])

        a_14 = a_41 = 1 / 3360 * (self.dzeta * (self.Eb / table["x"] ** 2) * (
                3 * table["Bi__"] - 2 * table["Bi__3"]) - 4 * (
                                          table["x"] ** 2 * table["K"] * (
                                          13 * table["fi1"] + table["fi2"])))  # !!!!! уТОЧНИТЬ ЗНАК

        a_22 = 1 / 1680 * (self.dzeta * (self.Eb / table["x"]) * (table["Bi__2"] +
                                                                    table["Bi__3"]) + 2 * table["x"] ** 3 *
                           table[
                               "K"] * (4 *
                                       table["fi1"] + table["fi2"]))

        a_23 = a_32 = 1 / 3360 * (-self.dzeta * (self.Eb / table["x"] ** 2) * (
                3 * table["Bi__"] + 2 * table["Bi__3"]) + 4 * table[
                                      "x"] ** 2 * table["K"] * (
                                          13 * table["fi1"] - table["fi2"]))

        a_24 = a_42 = 1 / 3360 * (
                self.dzeta * (self.Eb / table["x"]) * table["Bi__"] - 12 *
                table["x"] ** 3 * table["K"] * table["fi1"])

        a_33 = 1 / 560 * (self.dzeta * (self.Eb / table["x"] ** 3) * table["Bi__"] + 8 *
                          table["x"] * table["K"] * (
                                  13 * table["fi1"] - 7 * table["fi2"]))

        a_34 = a_43 = 1 / 3360 * (-self.dzeta * (self.Eb / table["x"] ** 2) * (
                3 * table["Bi__2"] - 2 * table["Bi__3"]) + 8 * table["x"] ** 2 * table["K"] * (
                                          4 * table["fi2"] - 11 * table["fi1"]))

        a_44 = 1 / 1680 * (self.dzeta * (self.Eb / table["x"]) * (table["Bi__2"] -
                                                                    table["Bi__3"]) + 2 * table["x"] ** 3 * table[
                               "K"] * (4 * table["fi1"] - table["fi2"]))

        R_11 = np.block([[a_11, a_12], [a_21, a_22]])
        print(f"R11= \n {R_11}")
        print(f" Размер массива {R_11.ndim}")
        R_22 = np.block([[a_33, a_34], [a_43, a_44]])
        print(f"R_22=\n{R_22} {R_22.shape}")

        R_12 = self.R_21 = np.block([[a_13, a_14], [a_23, a_24]])
        R_21 = self.R_21.transpose()
        print(f"R_12= \n{R_12}")

        print(f"cложен \n {R_22 + R_11}")

        k = np.block([[R_11, R_12], [R_21, R_22]])

        print(f" Размер массива \n k {k}  dim\n{k.ndim} \n shape{k.shape}")
        return k

    def fun_Matrix_u(self, matrix_force:np.array, matrix_B_gestk:np.array):
        """
        Матрица перемещений
        :return: матрица перемещений
        """

        print("---------+")
        print(matrix_force.shape, matrix_force.shape)
        print("=====-")

        U = np.linalg.solve(matrix_B_gestk, matrix_force)
        matrix_force=matrix_force
        #U = np.linalg.tensorsolve(matrix_B_gestk, matrix_force)
        return U


def table_iterrator(table: pd.DataFrame):
    for row in table.itertuples(index=True):
        print(f"--------------{row[0]}--------------------")
        data = SvAi(*(row[1:]))
        try:
            dictё_ = dict_.append(None, ignore_index=True)
        except:
            dict_ = pd.DataFrame(None, index=[0])


if __name__ == '__main__':
    book = xw.books

    # sheet = book.active.sheets
    wb = xw.Book(r'Расчет свай.xlsx')

    sheet_svai = wb.sheets["svai"]
    sheet_grunt = wb.sheets["grunt"]

    data_grunt = sheet_grunt.range("A1").options(pd.DataFrame, expand='table', index_col=True).value
    data_grunt["lsv"] = data_grunt["lsv"].apply(lambda x: x * 1000)  # Перевод в мм
    # data_grunt["ki"] = data_grunt.appply()

    data_svai = sheet_svai.range("A1").options(pd.DataFrame, expand='table', index_col=False).value  # Сваи
    data_svai: pd.DataFrame
    data_svai = data_svai.reset_index()

    table_iterrator(data_svai)  # Запуск построчной передачи в класс

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
