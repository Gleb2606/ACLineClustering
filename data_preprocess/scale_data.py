# Импорт необходимых библиотек
from enum import Enum
from sklearn.preprocessing import StandardScaler
from data_preprocess.data_load import *

class Float(Enum):
    """
    Перечисление видов чисел с плавающей точкой
    """
    FLOAT32 = np.float32
    FLOAT64 = np.float64

def data_scale(name: str, first_parameter: str, second_parameter: str,
               float_type: Float, samples: int) -> (pd.DataFrame, pd.DataFrame):
    """
    Функция нормализации данных в датафрейме
    :param name: Наименование файла
    :param first_parameter: Наименование первого параметра
    :param second_parameter: Наименование второго параметра
    :param float_type: Тип числа с плавающей точкой из перечисления
    :param samples: Количество данных в датафрейме (если 0 - то все)
    :return: Нормализованный и исходный DataFrame
    """
    # Загрузка датафрейма
    df_clean = data_load(name, first_parameter, second_parameter)

    # Сокращение данных в датафрейме (при необходимости)
    if samples != 0:
        df_clean = df_clean.sample(n=samples, random_state=42)

    # Инициализация алгоритма нормализации
    scaler = StandardScaler()

    # Нормализация данных
    X = scaler.fit_transform(df_clean[[f"{first_parameter}_clean", f"{second_parameter}_clean"]])
    X = X.astype(float_type.value)
    return X, df_clean
