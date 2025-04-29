# Импорт необходимых библиотек
from data_preprocess.read_csv import *
from data_preprocess.clean_column import *

def data_load(name: str, first_parameter: str, second_parameter: str) -> pd.DataFrame:
    """
    Функция загрузки csv файла, его преобразования в DataFrame и выделения данных для кластеризации
    :param name: Наименование csv файла
    :param first_parameter: Первый параметр кластеризации
    :param second_parameter: Второй параметр кластеризации
    :return: DataFrame с данными для кластеризации
    """
    # Загрузка данных
    df = read_csv(name)

    # Выделение столбцов, данные которых будут кластеризованы
    df[f'{first_parameter}_clean'] = clean_column(df[f'{first_parameter}'])
    df[f'{second_parameter}_clean'] = clean_column(df[f'{second_parameter}'])

    # Удаление пустых строк
    df_clean = df.dropna(subset=[f'{first_parameter}_clean', f'{second_parameter}_clean'])

    return df_clean