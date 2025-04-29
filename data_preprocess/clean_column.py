# Импорт необходимых библиотек
import numpy as np
import pandas as pd


def clean_column(column: pd.DataFrame) -> pd.Series:
    """
    Функция преобразования данных в столбце датафрейма
    :param column: Наименование столбца
    :return: Столбец обработанных данных
    """
    cleaned = []
    for value in column:
        if isinstance(value, str):
            #Удаление всех нецифровых символов, кроме точки
            cleaned_value = ''.join(filter(lambda x: x.isdigit() or x == '.', value))
            if cleaned_value:
                cleaned.append(float(cleaned_value))
            else:
                cleaned.append(np.nan)
        else:
            cleaned.append(value)
    return pd.Series(cleaned)