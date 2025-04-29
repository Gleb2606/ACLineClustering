# Импорт необходимых библиотек
import matplotlib.pyplot as plt
import pandas as pd

def plot(dataframe: pd.DataFrame, first_parameter: str, second_parameter: str,
         clusters: pd.DataFrame, method_name: str) -> None:
    """
    Функция построения графика распределения кластеризованных параметров
    :param dataframe: DataFrame с исходными параметрами
    :param first_parameter: Наименование первого параметра
    :param second_parameter: Наименование второго параметра
    :param clusters: DataFrame с метками кластеров
    :param method_name: Наименование алгоритма кластеризации
    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(dataframe[f'{first_parameter}_clean'], dataframe[f'{second_parameter}_clean'],
                          c=clusters, cmap='viridis', alpha=0.6)
    plt.xlabel(f'{first_parameter}')
    plt.ylabel(f'{second_parameter}')
    plt.title(f'Кластеризация {method_name}')
    plt.colorbar(scatter, label='Кластеры')
    plt.grid(True)
    plt.show()