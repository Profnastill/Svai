##
import math

import pandas as pd
import xlwings as xw

pd.set_option('display.max_columns', None)
# movies.head()


beton_type = ["В3,5", "В5", "В7,5", "В10", "В12,5", "В15", "В20", "В25", "В30", "В35", "В40", "В45", "В50", "В55",
              "В60", "В70", "B80", "B90", "В100"]

beton_Rbn = [2.7, 3.5, 5.5, 7.5, 9.5, 11, 15, 18.5, 22, 25.5, 29, 32, 36, 39.5, 43, 50, 57, 64, 71]

data_Rb = pd.DataFrame(index=beton_type, data=beton_Rbn, columns=["Rb"])

print(data_Rb)


class svai:
    def __int__(self, type_sv, P, M, N, l, b1, h1, b2, h2, Class_Bet, As, As_):
        """

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
        self.ln = 200  # Длина конечного элемента мм
        self.type_sv = type_sv
        self.P = P * 1000  # Перевод из кН в н
        self.M = M * 1000 * 100  # Н*мм
        self.N = N * 1000  # Продольная сила Н
        self.l = l  # длина сваи
        self.b1 = b1  # мм
        self.b2 = b2  # мм
        self.h1 = h1  # мм
        self.h2 = h2  # мм
        gamma_b = 1.3
        self.Rb = float(data_Rb[Class_Bet]["Rb"]) * gamma_b
        self.Es = 2 * 10 ** 5  # модуль упургости не напрягаемой арматуры
        self.Es_n = 2 * 10 ** 5  # модуль упругости напрягаемой арматуры !!!!!!!!!!!
        self.Eb = 30 * 10 ** 3  # модуль упругости бетона!!!!!!
        self.As = As * 100  # лощадь арматуры мм2
        self.As_ = As_ * 100  # площадь растянутой арматуры мм2


    def setka(self, data_grunt):
        """
        Разбивка сетки конечных элементов
        :return:
        """
        data_grunt: pd.DataFrame
        self.ln=200
        print(data_grunt)
        lis=[]
        for i in range(len(data_grunt)):
            print(i)
            if self.ln <= data_grunt.iloc[i]["lsv"]:
                print(data_grunt.iloc[i]["lsv"])
                col = data_grunt.iloc[i]["lsv"] // self.ln
                data_grunt.at[i+1,"lsv"]=self.ln
                lis.append(col)
                print(lis)

        data_grunt.loc.repeat(lis)
        print(data_grunt)


def test(data_svai: pd.DataFrame):
    data_svai = data_svai.apply(lambda x: data_svai)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    book = xw.books
    sheet = book.active.sheets

    sheet_svai = xw.sheets["svai"]
    sheet_grunt = xw.sheets["grunt"]

    data_grunt = sheet_grunt.range("A1").options(pd.DataFrame, expand='table', index_col=True).value
    data_grunt["lsv"] = data_grunt["lsv"].apply(lambda x: x * 1000)  # Перевод в мм

    sheet = sheet.active
    data_svai = sheet_svai.range("A1").options(pd.DataFrame, expand='table', index_col=True).value  # Сваи

    data_list: pd.DataFrame
    a = svai()

    a.setka(data_grunt)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
