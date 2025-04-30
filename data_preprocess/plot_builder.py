# Импорт необходимых библиотек
import matplotlib.pyplot as plt
import pandas as pd

def plot(dataframe: pd.DataFrame, parameters: list,
         clusters: pd.DataFrame, method_name: str) -> None:
    """
    Функция построения графика распределения кластеризованных параметров
    :param dataframe: DataFrame с исходными параметрами
    :param parameters: Список параметров для кластеризации
    :param clusters: DataFrame с метками кластеров
    :param method_name: Наименование алгоритма кластеризации
    """
    if len(parameters) == 2:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(dataframe[f'{parameters[0]}_clean'], dataframe[f'{parameters[1]}_clean'],
                              c=clusters, cmap='viridis', alpha=0.6)
        plt.xlabel(f'{parameters[0]}')
        plt.ylabel(f'{parameters[1]}')
        plt.title(f'Кластеризация {method_name}')
        plt.colorbar(scatter, label='Кластеры')
        plt.grid(True)
        plt.show()
