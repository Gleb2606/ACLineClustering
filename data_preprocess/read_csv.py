# Импорт необходимых библиотек
import pandas as pd

def read_csv(file_name: str) -> pd.DataFrame:
    """
    Функция загрузки csv файла
    :param file_name: Наименование файла
    :return: Датафрейм с данными из файла
    """
    return pd.read_csv(file_name, sep=';')