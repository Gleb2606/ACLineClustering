# Импорт необходимых библиотек
from sklearn.preprocessing import LabelEncoder
from data_preprocess.read_csv import *
from data_preprocess.clean_column import *

def data_load(name: str, parameters: list) -> (pd.DataFrame, list):
    """
    Функция загрузки csv файла, его преобразования в DataFrame и выделения данных для кластеризации
    :param name: Наименование файла
    :param parameters: Параметры для кластеризации
    :return: DataFrame после предобработки и набор параметров после предобработки
    """
    # Загрузка данных
    df = read_csv(name)

    # Инициализатор преобразователя категориальных параметров
    label_encoder = LabelEncoder()

    # подмножество параметров после предобработки
    subset_clean = []

    # Выделение столбцов, данные которых будут кластеризованы
    for parameter in parameters:
        if "(A)" in parameter:
            df[f'{parameter}_clean'] = label_encoder.fit_transform(df[f'{parameter}'])
        else:
            df[f'{parameter}_clean'] = clean_column(df[f'{parameter}'])

        subset_clean.append(f'{parameter}_clean')

    # Удаление пустых строк
    df_clean = df.dropna(subset=subset_clean)

    return df_clean, subset_clean