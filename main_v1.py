##
import math

import numpy
import numpy as np
import pandas as pd
import xlwings as xw

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 30
pd.options.display.expand_frame_repr = False
np.set_printoptions(edgeitems=30,linewidth=10000)
np.set_printoptions(edgeitems=4,linewidth=180)
np.set_printoptions(precision=3)

np.core.arrayprint._line_width = 80
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
        self.P = P * 1000  # Перевод из кН в н
        self.M = M * 1000 * 100  # Н*мм
        self.N = N * 1000  # Продольная сила Н
        self.l = l * 1000  # длина сваи мм
        self.b1 = b1  # мм
        self.b2 = b1  # мм
        self.h1 = h1  # мм
        self.h2 = h1  # мм
        gamma_b = 1.3
        print(data_Rb)
        self.k_zondir = (1 if Zondir == "Стабилизация" else 0.6)  # Коэффициент зондирования
        self.Rb = float(data_Rb["Rb"][Class_Bet]) * gamma_b
        self.Es = 2 * 10 ** 5  # модуль упургости не напрягаемой арматуры
        self.Es_n = 2 * 10 ** 5  # модуль упругости напрягаемой арматуры !!!!!!!!!!!
        self.Eb = 30 * 10 ** 3  # модуль упругости бетона!!!!!!
        self.Ea = 206000  # Н/мм2 модуль арматуры не напрягаемой
        self.Eaн = 200000  # Н/мм2 модуль арматуры напрягаемой
        self.As = As * 100  # лощадь арматуры мм2
        self.As_ = As_ * 100  # площадь растянутой арматуры мм2
        self.As2 = As2 * 100  # лощадь арматуры мм2
        self.As2_ = As2_ * 100  # площадь растянутой арматуры мм2
        self.a = 30  # Толщина защитного слоя    мм
        self.a_n = 40  # толщина защитного слоя для напрягаемой арматуры мм

        self.table = self.fun_Setka(data_grunt).copy()  # Разбиваем грунт на конечные элементы

        self.Ar_sum = self.table["Ar"].sum()

        self.table["x_1"] = self.table["lsv"].shift(1, fill_value=0).cumsum()  # ??
        self.table["x_2"] = self.table["lsv"].cumsum()  # ???

        self.table["fi1"] = self.b1 - (self.b1 - self.b2) / self.l * (
                self.table["x_1"] + self.table["x_2"])

        self.table["fi2"] = self.b1 - (self.b1 - self.b2) / self.l * (
                self.table["x_1"] - self.table["x_2"])

        self.table["bi"] = self.table["fi1"] / 2
        self.fun_Bi()#Жесткость элемента

        print(self.table)

        self.table["K"] = self.table.apply(lambda row: self.Koeffic_Postely(row), axis=1)#определяем коэффициент постели
        self.table["B"]=self.table.apply(lambda row: self.matrix_B_piramida(row), axis=1)#Матрица жесткости
        self.len_matr = len(self.table)
        matrix_force = self.fun_Matrix_Force()
        print(self.table["B"])
        self.table["U"]=self.table["B"].apply(lambda row: self.fun_matrix_u(matrix_force, row))

        print(self.table)


        """""""""
        self.data_grunt_new["K"] = self.data_grunt_new.apply(lambda row: self.Koeffic_Postely(row), axis=1)
       # self.data_grunt_new["B"] = self.data_grunt_new.apply(lambda row: self.Gest_Sechen(row),
       #                                                      axis=1)  # Жесткость элемента

        self.data_grunt_new["k_elem"] = self.data_grunt_new.apply(lambda row: self.matrix_B(row),

                                                                  axis=1)  # Матрица жесткости
                                                                  """""


        # print(self.data_grunt_new)

    def fun_Bi(self):
        """
        Определение коэффициентов
        :return:
        """

        self.table['Bi__'] = (35 * self.table["fi1"] ** 4) + (
                126 * self.table["fi1"] ** 2 + self.table["fi2"] ** 2) + (
                                     15 * self.table["fi2"] ** 4)
        self.table['Bi__1'] = 35 * self.table["fi1"] ** 4 + (
                154 * self.table["fi1"] ** 2 + self.table["fi2"] ** 2) + (
                                      19 * self.table["fi2"] ** 4)
        self.table['Bi__2'] = (35 * self.table["fi1"] ** 4) + (
                112 * self.table["fi1"] ** 2 + self.table["fi2"] ** 2) + (
                                      13 * self.table["fi2"] ** 4)
        self.table['Bi__3'] = (70 * self.table["fi1"] ** 4) + (
                42 * self.table["fi1"] ** 2 + self.table["fi2"] ** 3)

    def fun_Setka(self, data_grunt):
        """
        Разбивка сетки конечных элементов
        :return:
        """
        data_grunt: pd.DataFrame
        print(data_grunt)
        lis = []
        print("dfs")
        for i in range(len(data_grunt)):
            if self.ln_elem <= data_grunt.iloc[i]["lsv"]:
                col = data_grunt.iloc[i]["lsv"] // self.ln_elem
                data_grunt.at[i + 1, "lsv"] = self.ln_elem
                lis.append(col)

        data_grunt = data_grunt.loc[data_grunt.index.repeat(lis)].reset_index(drop=True)
        data_grunt["sumLen"] = data_grunt["lsv"].cumsum()
        data_grunt = data_grunt.query("sumLen<= @self.l")
        return data_grunt

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
        matrix_force = np.append(matrix_force, np.zeros((self.len_matr * 2 - 2, 1)))
        print("-----------")
        print(type(matrix_force))
        return matrix_force

    def matrix_B_piramida(self, table: pd.DataFrame):
        """
        Матрица жесткостей
        :param table:
        :return: Матрица жесткости
        """
        #print(table)
       # I = self.Moment_inerc(table["b"], table["h"])
        self.dzeta = 1
        a_11 = 1 / 560 * (self.dzeta * ((self.Eb / table["lsv"] ** 3) * table["Bi__"]) + 8 *
                          table["lsv"] * table["K"] * (
                                  13 * table["fi1"] + 7 * table["fi2"]))

        a_12 = a_21 = 1 / 3360 * (self.dzeta * ((self.Eb / table["lsv"] ** 2) * (
                3 * table["Bi__"] + 2 * table["Bi__3"])) + 8 * (table[
                                                                                               "lsv"] ** 2) *
                                  table["K"] * (
                                          11 * table["fi1"] + 4 * table["fi2"]))
        a_13 = a_31 = 1 / 560 * (
                -1 * self.dzeta * (self.Eb / table["lsv"] ** 3) * table["Bi__"] + 36 *
                table["lsv"] * table["K"] * table["fi1"])

        a_14 = a_41 = 1 / 3360 * (self.dzeta * (self.Eb / table["lsv"] ** 2) * (
                3 * table["Bi__"] - 2 * table["Bi__3"]) - 4 * (
                                          table["lsv"] ** 2 * table["K"] * (
                                          13 * table["fi1"] + table["fi2"])))# !!!!! уТОЧНИТЬ ЗНАК

        a_22 = 1 / 1680 * (self.dzeta * (self.Eb / table["lsv"]) * (table["Bi__2"] +
                                                                         table["Bi__3"]) + 2 * table["lsv"] ** 3 *
                           table[
                               "K"] * (4 *
                                       table["fi1"] + table["fi2"]))

        a_23 = a_32 = 1 / 3360 * (-self.dzeta * (self.Eb / table["lsv"] ** 2) * (
                3 * table["Bi__"] + 2 * table["Bi__3"]) + 4 * table[
                                      "lsv"] ** 2 * table["K"] * (
                                          13 * table["fi1"] - table["fi2"]))

        a_24 = a_42 = 1 / 3360 * (
                self.dzeta * (self.Eb / table["lsv"]) * table["Bi__"] - 12 *
                table["lsv"] ** 3 * table["K"] * table["fi1"])

        a_33 = 1 / 560 * (self.dzeta * (self.Eb / table["lsv"] ** 3) * table["Bi__"] + 8 *
                          table["lsv"] * table["K"] * (
                                  13 * table["fi1"] - 7 * table["fi2"]))

        a_34 = a_43 = 1 / 3360 * (-self.dzeta * (self.Eb / table["lsv"] ** 2) * (
                3 * table["Bi__2"] - 2 * table["Bi__3"]) + 8 * table["lsv"] ** 2 * table["K"] * (
                                          4 * table["fi2"] - 11 * table["fi1"]))

        a_44 = 1 / 1680 * (self.dzeta * (self.Eb / table["lsv"]) * (table["Bi__2"] -
                                                                         table["Bi__3"]) + 2 * table["lsv"] ** 3 * table[
                   "K"] * (4 * table["fi1"] - table["fi2"]))

        k_elem = np.array([[a_11, a_12, a_13, a_14],
                           [a_21, a_22, a_23, a_24],
                           [a_31, a_32, a_33, a_34],
                           [a_41, a_42, a_43, a_44]])  # self.ln лина конечного элемента
        print(k_elem)
        print(k_elem.shape)
        return k_elem  # Матрица жесткости

    def fun_matrix_u(self,matrix_force,matrix_B_gestk):
        """
        Матрица перемещений
        :return: матрица перемещений


        """

        U:np
        print("---------+")
        print(type(matrix_force),len(matrix_force))
        print(type(matrix_B_gestk),len(matrix_B_gestk))
        print(matrix_B_gestk.shape)
        print(matrix_B_gestk)


        print("=====-")

        U=np.dot(np.linalg.matrix_power(matrix_B_gestk, -1)*matrix_force)


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

    sheet = book.active.sheets

    sheet_svai = xw.sheets["svai"]
    sheet_grunt = xw.sheets["grunt"]

    data_grunt = sheet_grunt.range("A1").options(pd.DataFrame, expand='table', index_col=True).value
    data_grunt["lsv"] = data_grunt["lsv"].apply(lambda x: x * 1000)  # Перевод в мм
    # data_grunt["ki"] = data_grunt.appply()

    sheet = sheet.active
    data_svai = sheet_svai.range("A1").options(pd.DataFrame, expand='table', index_col=False).value  # Сваи
    data_svai: pd.DataFrame
    data_svai = data_svai.reset_index()

    table_iterrator(data_svai)  # Запуск построчной передачи в класс

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
