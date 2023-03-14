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
        self.ln_elem = 10  # Коэфициент разбивки на КЭ
        self.type_sv = type_sv
        self.P = P * 1000  # 00000000000000000000  # Перевод из кН в н
        self.M = M * 1000 * 100  # Н*мм
        self.N = N * 1000  # Продольная сила Н
        self.lsv = l * 1000  # длина сваи мм
        self.b1 = b1  # мм
        self.b2 = b1  # мм
        self.h1 = h1  # мм
        self.h2 = h1  # мм
        gamma_b = 1.3
        self.k_zondir = (1 if Zondir == "Стабилизация" else 0.6)  # Коэффициент зондирования
        self.Rb = float(data_Rb["Rb"][Class_Bet]) * gamma_b
        self.Eb = 3 * 10 ** 4  # модуль упругости бетона Н/мм2 МПа!!!!!!

        self.Ea = 206000  # Н/мм2 модуль арматуры не напрягаемой
        self.As = As * 10  # лощадь арматуры мм2
        self.As_ = As_ * 100  # площадь растянутой арматуры мм2
        self.As2 = As2 * 100  # лощадь арматуры мм2
        self.As2_ = As2_ * 100  # площадь растянутой арматуры мм2
        self.a = 30  # Толщина защитного слоя    мм
        self.a_n = 40  # толщина защитного слоя для напрягаемой арматуры мм

        self.table_bs = self.fun_Setka(data_grunt).copy()  # Разбиваем грунт на конечные элементы
        self.Ar_sum = self.table_bs["Ar"].sum()

        self.fun_Bi()  # Жесткость элемента

        #print(self.table_bs)

        self.len_matr = len(self.table_bs)  # Количество конечных элементов

        self.table_bs["C"] = self.table_bs.apply(lambda row: self.Koeffic_Postely(row),  # В работе обозначен как К
                                                 axis=1)  # определяем коэффициент постели
        self.table_bs["B"] = self.table_bs.apply(lambda row: self.matrix_B_piramida(row), axis=1)  # Матрица жесткости

        # self.table_bs["B"] = self.table_bs.apply(lambda row: self.fun_Bi_kvadrat(row), axis=1)  # Матрица жесткоcти для квадратной

        K_ob = self.MaTrix()  # Получение диагональной  обобщенной матрицы жесткости
        matrix_force = self.fun_Matrix_Force()
        u = self.fun_Matrix_u(matrix_force, K_ob)  # Матрица перемещений общая
        print(f"Перемещения в узлах от обобщенном матрицы\n {u}")
        self._usilia(u)#Усилия в каждом конечном элементе


    def _usilia(self,u:numpy):
        """
        Определяются перемещения и усилия в середине каждого конечного элемента

        :param u: матрица с перемещениями и поворотами
        :return: self.table Таблица итоговая
        """
        fi = u[1::2]
        u = u[::2]

        u_sum = u[:-1:] + u[1::]  # Перемещения
        u_diff = np.diff(u)

        fi_diff = np.diff(fi)# Разность
        fi_sum = fi[:-1:] + fi[1::]

        self.table_bs["u_мм"] = 1 / 2 * (u_sum) + 1 / 8 * self.table_bs["lкэл"] * fi_sum

        self.table_bs["fi_"] = 3 / (2 * self.table_bs["lкэл"]) * fi_diff - 1 / 4 * fi_sum

        self.table_bs["QкН"] = (6 * self.Gest_Sechen_EI(self.table_bs) / (self.table_bs["lкэл"] ** 3) * (
                    2 * u_diff - self.table_bs["lкэл"] * fi_sum))/1000
        self.table_bs["MкН*м"]=(self.Gest_Sechen_EI(self.table_bs) / (self.table_bs["lкэл"])* fi_diff)/1000000

        self.table_bs.drop(
            ['x', "x_1", "x_2", "b1i", "b2i", "bi", "fi1", "fi2", "Bi__", "Bi__1", "Bi__2", "Bi__3", "B", "K_ob"], axis=1, inplace=True)

        return self.table_bs


    def MaTrix(self):
        """
        Получение обобщенной диагональной  матрицы жестскости
        :return:numpy array  Обобщенная матрица
        """
        print("-------------")
        self.table_bs["K_ob"] = 0

        mr = matrix_reshen.Matrix(self.len_matr)

        K_ob = mr.nul_mat
        for i in self.table_bs.index:
            K_ob += mr.mat_umn(self.table_bs.iloc[i]["B"], i)  # получение обобщенной матрицы для каждой строки
            # print(f"----Матрица ---{i} \n   {K_ob} ")
            mr = matrix_reshen.Matrix(
                self.len_matr)  # Перезапускаем класс матрицы чтобы начальаная была нулевая матрица иначе почемеу то получается завдвоение значений

        print(f"----Матрица --- \n   {K_ob} ")

        return K_ob

    def fun_Bi(self):
        """
        Определение коэффициентов
        :return:
        """

        self.table_bs["x_1"] = self.table_bs["x"]
        self.table_bs["x_2"] = self.table_bs["x"].shift(-1, fill_value=0)

        self.table_bs["lкэл"] = self.table_bs["x_2"] - self.table_bs["x_1"]
        self.table_bs.drop(self.table_bs.index[-1], inplace=True)  # Удаляем последнюю строку

        self.table_bs["b1i"] = self.b1 - (self.b1 - self.b2) / self.lsv * (
            self.table_bs["x_1"])

        self.table_bs["b2i"] = self.b1 - (self.b1 - self.b2) / self.lsv * (
            self.table_bs["x_2"])

        self.table_bs["bi"] = (self.table_bs["b1i"] + self.table_bs["b2i"]) / 2

        self.table_bs["fi1"] = 2 * self.table_bs["b1i"]
        self.table_bs["fi2"] = self.table_bs["b1i"] - self.table_bs["b2i"]  # Тут получился 0?

        self.table_bs['Bi__'] = (35 * self.table_bs["fi1"] ** 4) + (
                126 * self.table_bs["fi1"] ** 2) * (self.table_bs["fi2"] ** 2) + (
                                        15 * self.table_bs["fi2"] ** 4)
        self.table_bs['Bi__1'] = (35 * self.table_bs["fi1"] ** 4) + (
                154 * self.table_bs["fi1"] ** 2) * (self.table_bs["fi2"] ** 2) + (
                                         19 * self.table_bs["fi2"] ** 4)
        self.table_bs['Bi__2'] = (35 * self.table_bs["fi1"] ** 4) + (
                112 * self.table_bs["fi1"] ** 2) * (self.table_bs["fi2"] ** 2) + (
                                         13 * self.table_bs["fi2"] ** 4)
        self.table_bs['Bi__3'] = (70 * self.table_bs["fi1"] ** 3) * (self.table_bs["fi2"]) + (
                42 * self.table_bs["fi1"] * self.table_bs["fi2"] ** 3)

        self.table_bs

    def Moment_inerc(self, a, b):
        """
        момент инеруии
        :return:
        """

        I = (a * b ** 3) / 12
        return I

    def fun_Bi_kvadrat(self, table: pd.DataFrame):
        """
        Матрица жесткостей для прямоугольной сваи
        :param table:
        :return: Матрица жесткости
        """
        # print(table)
        table["b"] = self.b1
        table["h"] = self.h1

        self.ln_elem = table["lкэл"]
        I = self.Moment_inerc(table["b"], table["h"])

        a_11 = 13 / 35 * table["C"] * table["b"] * self.ln_elem + 12 * self.Eb * I / (self.ln_elem ** 3)
        a_12 = a_21 = 11 / 210 * table["C"] * table["b"] * self.ln_elem ** 2 + 6 * self.Eb * I / (self.ln_elem ** 2)
        a_13 = a_31 = 9 / 70 * table["C"] * table["b"] * self.ln_elem - 12 * self.Eb * I / (self.ln_elem ** 3)
        a_14 = a_41 = 6 * self.Eb * I / (self.ln_elem ** 2) - 13 / 420 * table["C"] * table["b"] * self.ln_elem ** 2
        a_22 = 1 / 105 * table["C"] * table["b"] * self.ln_elem ** 3 + 4 * self.Eb * I / self.ln_elem
        a_23 = a_32 = 13 / 420 * table["C"] * table["b"] * self.ln_elem ** 2 - 6 * self.Eb * I / self.ln_elem ** 2
        a_24 = a_42 = 2 * self.Eb * I / self.ln_elem - 1 / 140 * table["C"] * table["b"] * self.ln_elem ** 3
        a_33 = 13 / 35 * table["C"] * table["b"] * self.ln_elem + 12 * self.Eb * I / self.ln_elem ** 3
        a_34 = a_43 = -11 / 210 * table["C"] * table["b"] * self.ln_elem ** 2 - 6 * self.Eb * I / self.ln_elem ** 2
        a_44 = 1 / 105 * table["C"] * table["b"] * self.ln_elem ** 3 + 4 * self.Eb * I / self.ln_elem

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

    def fun_Setka(self, data_grunt):
        """
        Разбивка сетки конечных элементов
        :return:
        """
        data_grunt: pd.DataFrame
        print(data_grunt)
        lis = []
        da = pd.DataFrame(columns=data_grunt.columns)
        last = 0
        for i in range(len(data_grunt)):
            # self.ln_elem = 10
            col = np.linspace(0, data_grunt.iloc[i]["lsloy"], self.ln_elem)

            col = col.tolist()
            col = list(map(lambda x: x + last, col))

            f = [data_grunt.iloc[i]] * (len(col) - 1)  # Размножаем строку таблицы грунта
            da = da.append(f, ignore_index=True)
            lis.extend(col)  # Список координат свай

            last = lis[-1]
            lis.pop()

        da = da.append([data_grunt.iloc[-1]], ignore_index=True)
        lis.extend([last])
        da["x"] = lis  # Координаты начала и конца конечного элемента

        da = da.query("x<=@self.lsv")
        if da.tail(n=1)["x"].values < self.lsv:
            da = pd.concat([da, da.tail(1)], ignore_index=True)
            da.at[int(da.tail(1).index.values), "x"] = self.lsv

        da = da.reindex()



        return da

    def Koeffic_Postely(self, table: pd.DataFrame):
        """
        :param table:
        :return: коэффициент постели
        """
        print(table)
        print("----")
        print(f"l={self.lsv},{table['bi']}")
        self.w = self.lsv // table["bi"]
        print("-----------w= ", self.w)
        if 4 <= self.w <= 10:
            self.w = data_w["lc_bi"][self.w]
        elif self.w < 4:
            self.w = data_w["lc_bi"][4]
        elif self.w >= 10:
            self.w = data_w["lc_bi"][10]
        print("w=", self.w)

        if table["Er"] != None and table["Er"] != 0:
            k = table["Ar"] * table["Er"] * self.w / ((1 - table["Nu"] ** 2) * table["bi"])
        else:
            k = self.k_zondir * table["Ar"] * self.w / ((1 - table["Nu"] ** 2) * table["bi"]) * (
                    5.5 * table["qi"] + (table["qi"] / 50) ** 3)  # тут в кубе?

        return k

    def Gest_Sechen_EI(self, table: pd.DataFrame):
        """
        Жесткость элемента
        :return:
        """
        na = self.Ea / self.Eb
        nn = self.Ea / self.Eb

        EI = ((self.table_bs["bi"] ** 4) / 12 + (
                (na * self.As * (self.table_bs["bi"]) / 2 - self.a) ** 2 + nn * self.As2 * (
            self.table_bs["bi"] / 2 - self.a) ** 2)) * 0.85 * self.Eb

        #EI=(self.table_bs["bi"] ** 4)*self.Eb
        return EI

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
        a_11 = 1 / 560 * (self.dzeta * ((self.Eb / table["lкэл"] ** 3) * table["Bi__"]) + 8 *
                          table["lкэл"] * table["C"] * (
                                  13 * table["fi1"] + 7 * table["fi2"]))

        a_12 = a_21 = 1 / 3360 * (self.dzeta * ((self.Eb / table["lкэл"] ** 2) * (
                3 * table["Bi__"] + 2 * table["Bi__3"])) + 8 * (table[
                                                                    "lкэл"] ** 2) *
                                  table["C"] * (
                                          11 * table["fi1"] + 4 * table["fi2"]))

        test = self.dzeta * ((self.Eb / table["lкэл"] ** 2) * (3 * table["Bi__"] + 2 * table["Bi__3"])) / (
                    8 * (table["lкэл"] ** 2) * table["C"] * (11 * table["fi1"] + 4 * table["fi2"]))

        a_13 = a_31 = 1 / 560 * (
                -1 * self.dzeta * (self.Eb / table["lкэл"] ** 3) * table["Bi__"] + 36 *
                table["lкэл"] * table["C"] * table["fi1"])

        a_14 = a_41 = 1 / 3360 * (self.dzeta * (self.Eb / table["lкэл"] ** 2) * (
                3 * table["Bi__"] - 2 * table["Bi__3"]) - 4 * (
                                          table["lкэл"] ** 2 * table["C"] * (
                                          13 * table["fi1"] + table["fi2"])))  # !!!!! уТОЧНИТЬ ЗНАК

        a_22 = 1 / 1680 * (self.dzeta * (self.Eb / table["lкэл"]) * (table["Bi__2"] +
                                                                     table["Bi__3"]) + 2 * table["lкэл"] ** 3 *
                           table[
                               "C"] * (4 *
                                       table["fi1"] + table["fi2"]))

        a_23 = a_32 = 1 / 3360 * (-self.dzeta * (self.Eb / table["lкэл"] ** 2) * (
                3 * table["Bi__"] + 2 * table["Bi__3"]) + 4 * table[
                                      "lкэл"] ** 2 * table["C"] * (
                                          13 * table["fi1"] - table["fi2"]))

        a_24 = a_42 = 1 / 3360 * (
                self.dzeta * (self.Eb / table["lкэл"]) * table["Bi__"] - 12 *
                (table["lкэл"] ** 3) * table["C"] * table["fi1"])

        a_33 = 1 / 560 * (self.dzeta * (self.Eb / table["lкэл"] ** 3) * table["Bi__"] + 8 *
                          table["lкэл"] * table["C"] * (
                                  13 * table["fi1"] - 7 * table["fi2"]))

        a_34 = a_43 = 1 / 3360 * (-self.dzeta * (self.Eb / table["lкэл"] ** 2) * (
                3 * table["Bi__2"] - 2 * table["Bi__3"]) + 8 * table["lкэл"] ** 2 * table["C"] * (
                                          4 * table["fi2"] - 11 * table["fi1"]))

        a_44 = 1 / 1680 * (self.dzeta * (self.Eb / table["lкэл"]) * (table["Bi__2"] -
                                                                     table["Bi__3"]) + 2 * (table["lкэл"] ** 3) * table[
                               "C"] * (4 * table["fi1"] - table["fi2"]))

        R_11 = np.block([[a_11, a_12], [a_21, a_22]])  # Тут какая то проблема
        #print(f"R11= \n {R_11}")
        #print(f" Размер массива {R_11.ndim}")
        R_22 = np.block([[a_33, a_34], [a_43, a_44]])
        #print(f"R_22=\n{R_22} {R_22.shape}")

        R_12 = np.block([[a_13, a_14], [a_23, a_24]])
        R_21 = R_12.transpose()
        #print(f"R_12= \n{R_12}")
        #print(f"cложен \n {R_22 + R_11}")

        k = np.block([[R_11, R_12], [R_21, R_22]])
        #print(f" Размер массива \n k {k}  dim\n{k.ndim} \n shape{k.shape}")
        return k

    def fun_Matrix_u(self, matrix_force: np.array, matrix_B_gestk: np.array):
        """
        Матрица перемещений
        :return: матрица перемещений
        """
        U = np.linalg.solve(matrix_B_gestk, matrix_force)
        return U


def table_iterrator(table: pd.DataFrame):
    for row in table.itertuples(index=True):
        print(f"--------------{row[0]}--------------------")
        data = SvAi(*(row[1:]))
        print(data.table_bs)
        try:
            dict_ = dict_.append(None, ignore_index=True)
        except:
            dict_ = pd.DataFrame(None, index=[0])


if __name__ == '__main__':
    book = xw.books

    # sheet = book.active.sheets
    wb = xw.Book(r'Расчет свай.xlsx')

    sheet_svai = wb.sheets["svai"]
    sheet_grunt = wb.sheets["grunt"]

    data_grunt = sheet_grunt.range("A1").options(pd.DataFrame, expand='table', index_col=True).value
    data_grunt["lsloy"] = data_grunt["lsloy"].apply(lambda x: x * 1000)  # Перевод в  длины слоя мм

    data_svai = sheet_svai.range("A1").options(pd.DataFrame, expand='table', index_col=False).value  # Сваи
    data_svai: pd.DataFrame
    data_svai = data_svai.reset_index()

    table_iterrator(data_svai)  # Запуск построчной передачи в класс

