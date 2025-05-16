# Импорт необходимых библиотек
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from data_preprocess.scale_data import data_scale, Float

class BaseClustering(ABC):
    """
    Абстрактный класс алгоритм кластеризации
    """
    def __init__(self, file_path: str):
        """
        Конструктор класса
        :param file_path: Путь к файлу
        """
        self.file_path = file_path
        self.original_parameters = []
        self.X = None
        self.df_raw = None
        self.df_clean = None
        self.clusters = None

    @abstractmethod
    def perform_clustering(self, parameters: list) -> None:
        """
        Абстрактный метод выполнение кластеризации
        :param parameters: Список параметров кластеризации
        """
        pass

    def prepare_data(self, parameters: list) -> None:
        """
        Метод подготовки и нормализации данных с очисткой предыдущих результатов
        :param parameters: Список параметров кластеризации
        """
        self.X = None
        self.df_clean = None
        self.clusters = None
        self.df_raw = pd.read_csv(self.file_path, sep=';')
        self.X, self.df_clean = data_scale(
            self.file_path,
            parameters,
            Float.FLOAT32,
            0
        )
        self.original_parameters = parameters

    def get_statistics(self) -> str:
        """
        Метод получения статистики по кластерам
        :return: Статистика по кластерам
        """
        if self.clusters is None:
            return "Кластеризация не выполнена"

        stats = []
        unique_clusters, counts = np.unique(self.clusters, return_counts=True)
        stats.append("Статистика кластеров:\n")
        for cluster, count in zip(unique_clusters, counts):
            stats.append(f"Кластер {cluster} имеет {count} точек\n")

        valid_clusters = [c for c in unique_clusters if c != -1]
        for cluster in valid_clusters:
            cluster_points = self.df_clean[self.clusters == cluster][
                [f'{self.original_parameters[0]}_clean',
                 f'{self.original_parameters[1]}_clean']
            ]
            center = cluster_points.mean().values
            stats.append(f"Центр кластера {cluster}: ({center[0]:.2f}, {center[1]:.2f})\n")

        return "".join(stats)

    def get_plot_data(self) -> dict:
        """
        Метод получения данных для построения графиков
        :return: Данные для построения графиков
        """
        return {
            'df_clean': self.df_clean,
            'parameters': self.original_parameters,
            'clusters': self.clusters
        }

    def get_cluster_data(self) -> pd.DataFrame:
        """
        Метод получения информации по кластерам
        :return: Столбец с наименованиями кластеров
        """
        result_df = self.df_raw.copy()
        result_df['кластер'] = np.nan
        result_df.loc[self.df_clean.index, 'кластер'] = self.clusters
        return result_df