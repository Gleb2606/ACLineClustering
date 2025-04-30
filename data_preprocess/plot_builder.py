# Импорт необходимых библиотек
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    fig = plt.figure(figsize=(6, 4), dpi=100)

    if len(parameters) == 2:
        scatter = plt.scatter(dataframe[f'{parameters[0]}_clean'],
                              dataframe[f'{parameters[1]}_clean'],
                              c=clusters, cmap='viridis', alpha=0.6)
        plt.xlabel(f'{parameters[0]}')
        plt.ylabel(f'{parameters[1]}')
        plt.title(f'Кластеризация {method_name}')
        plt.colorbar(scatter, label='Кластеры')
        plt.grid(True)
        plt.show()

    elif len(parameters) == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(dataframe[f'{parameters[0]}_clean'],
                   dataframe[f'{parameters[1]}_clean'],
                   dataframe[f'{parameters[2]}_clean'],
                   c=clusters, cmap='viridis')
        ax.set_xlabel(f'{parameters[0]}')
        ax.set_ylabel(f'{parameters[1]}')
        ax.set_zlabel(f'{parameters[2]}')
        ax.set_title(f'Кластеризация {method_name}')
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Кластеры')
        fig.show()

