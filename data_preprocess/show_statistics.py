# Импорт необходимых библиотек
import numpy as np
import pandas as pd

def show_statistics(dataframe: pd.DataFrame, parameters: list, clusters: pd.DataFrame) -> None:
    """
    Функция сбора статистики по кластерам
    :param dataframe: DataFrame с данными
    :param parameters: Список параметров для кластеризации
    :param clusters: DataFrame с метками кластеров
    """
    # Статистика по кластерам
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    print("Статистика кластеров:")
    for cluster, count in zip(unique_clusters, counts):
        print(f"Кластер {cluster} имеет {count} точек")

    # Вывод центров кластеров
    valid_clusters = [c for c in unique_clusters if c != -1]
    for cluster in valid_clusters:
        cluster_points = dataframe[clusters == cluster][[f'{parameters[0]}_clean', f'{parameters[1]}_clean']]
        center = cluster_points.mean().values
        print(f"Центр кластера {cluster}: ({center[0]:.2f}, {center[1]:.2f})")