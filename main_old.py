##
import math

import numpy as np
import pandas as pd
import xlwings as xw
import matrix_reshen as mt

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20
pd.options.display.expand_frame_repr = False
# movies.head()


mt=mt.Matrix()

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
        self.b2 = b2  # мм
        self.h1 = h1  # мм
        self.h2 = h2  # мм
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

        self.data_grunt_new = self.setka(data_grunt).copy()  # Разбиваем грунт на конечные элементы

        self.Ar_sum = self.data_grunt_new["Ar"].sum()

        self.data_grunt_new["b"] = self.data_grunt_new["sumLen"] * (self.b1 - self.b2) / self.l + self.b2
        self.data_grunt_new["b"] = (self.data_grunt_new["b"] + self.data_grunt_new["b"].shift(-1,
                                                                                              fill_value=self.b1)) / 2# Находим сечение по центру конечного элемента
        self.data_grunt_new["h"] = self.data_grunt_new["sumLen"] * (self.h1 - self.h2) / self.l + self.h2
        self.data_grunt_new["h"] = (self.data_grunt_new["h"] + self.data_grunt_new["h"].shift(-1,
                                                                                              fill_value=self.h1)) / 2# Находим сечение по центру конечного элемента

        self.data_grunt_new["k"] = self.data_grunt_new.apply(lambda row: self.koeffic_postely(row), axis=1)
        self.data_grunt_new["B"] = self.data_grunt_new.apply(lambda row: self.Gest_Sechen(row),
                                                             axis=1)  # Жесткость элемента
        self.data_grunt_new=self.data_grunt_new.drop(labels=[0, len(self.data_grunt_new)-1]).reindex()#удаляем из таблицы первый и последний элемент, так как это не кэ а просто край

        print(self.data_grunt_new)
        self.data_grunt_new["k_elem"] = self.data_grunt_new.apply(lambda row: self.matrix_B(row),

                                                                  axis=1)  # Матрица жесткости
        self.len_matr = len(self.data_grunt_new)
        self.matrix_force = self.Matrix_Force()
        self.data_grunt_new["u"]=0# заполняем перемещения 0

        # print(self.data_grunt_new)

    def setka(self, data_grunt):
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

    def koeffic_postely(self, table: pd.DataFrame):
        """
        :param table:
        :return: коэффициент постели
        """
        print(table)
        print("----")
        print(f"l={self.l},{table['b']}")
        self.w = self.l // table["b"]
        print("-----------w= ", self.w)
        if self.w < 10:
            self.w = data_w.query("lc_bi==@self.w")
        elif self.w >= 10:
            self.w = data_w["lc_bi"][10]
        print("w=", self.w)

        if table["Er"] != None and table["Er"] != 0:
            k = table["Ar"] * table["Er"] * self.w / (1 - table["Nu"] ** 2) * table["b"]
        else:
            k = self.k_zondir * self.Ar_sum * self.w / ((1 - table["Nu"] ** 2)) * table["b"] * (
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

    def Moment_inerc(self, a, b):
        """
        момент инеруии
        :return:
        """

        I = (a * b ** 3) / 12
        return I

    def Matrix_Force(self):
        matrix_force = np.array([[self.P], [self.M]])
        matrix_force = np.append(matrix_force, np.zeros((self.len_matr - 2, 1)))
        print("-----------")
        print(matrix_force)

        return matrix_force

    def matrix_B(self, table: pd.DataFrame):
        """
        Матрица жесткостей
        :param table:
        :return: Матрица жесткости
        """
        # print(table)
        I = self.Moment_inerc(table["b"], table["h"])

        a_11 = 13 / 35 * table["k"] * table["b"] * self.ln_elem + 12 * self.Eb * I / (self.ln_elem ** 3)
        a_12 = a_21 = 11 / 210 * table["k"] * table["b"] * self.ln_elem ** 2 + 6 * self.Eb * I / (self.ln_elem ** 2)
        a_13 = a_31 = 9 / 70 * table["k"] * table["b"] * self.ln_elem - 12 * self.Eb * I / (self.ln_elem ** 3)
        a_14 = a_41 = 6 * self.Eb * I / (self.ln_elem ** 2) - 13 / 420 * table["k"] * table["b"] * self.ln_elem ** 2
        a_22 = 1 / 105 * table["k"] * table["b"] * self.ln_elem ** 3 + 4 * self.Eb * I / self.ln_elem
        a_23 = a_32 = 13 / 420 * table["k"] * table["b"] * self.ln_elem ** 2 - 6 * self.Eb * I / self.ln_elem ** 2
        a_24 = a_42 = 2 * self.Eb * I / self.ln_elem - 1 / 140 * table["k"] * table["b"] * self.ln_elem ** 3
        a_33 = 13 / 35 * table["k"] * table["b"] * self.ln_elem + 12 * self.Eb * I / self.ln_elem ** 3
        a_34 = a_43 = -11 / 210 * table["k"] * table["b"] * self.ln_elem ** 2 - 6 * self.Eb * I / self.ln_elem ** 2
        a_44 = 1 / 105 * table["k"] * table["b"] * self.ln_elem ** 3 + 4 * self.Eb * I / self.ln_elem






        k_elem = np.array([[a_11, a_12, a_13, a_14], [a_21, a_22, a_23, a_24], [a_31, a_32, a_33, a_34],
                           [a_41, a_42, a_43, a_44]])  # self.ln лина конечного элемента
        return k_elem  # Матрица жесткости

    def matrix_u(self):
        """
        Матрица перемещений
        :return:
        """
        return


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
