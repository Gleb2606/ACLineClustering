# Импорт необходимых библиотек
import pandas as pd
import numpy as np
from clustering_methods.Base_clustering import BaseClustering
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

class DBSCANClustering(BaseClustering):
    """
    Класс кластеризации методом DBSCAN
    """
    def __init__(self, file_path: str):
        """
        Конструктор класса
        :param file_path: Путь к файлу
        """
        super().__init__(file_path)
        self.optimal_eps = None
        self.min_samples = None

    def calculate_hyperparameters(self) -> None:
        """
        Автоматический расчет оптимальных гиперпараметров
        """
        self.min_samples = len(self.original_parameters) * self.X.shape[1]
        neigh = NearestNeighbors(n_neighbors=self.min_samples)
        nbrs = neigh.fit(self.X)
        distances, _ = nbrs.kneighbors(self.X)
        k_distances = np.sort(distances[:, self.min_samples - 1])
        differences = k_distances[1:] - k_distances[:-1]
        max_diff_idx = np.argmax(differences)
        self.optimal_eps = k_distances[max_diff_idx]

    def perform_clustering(self, parameters: list) -> pd.DataFrame:
        """
        Выполнение кластеризации DBSCAN
        :param parameters: Список параметров кластеризации
        :return: Датафрейм сформированных кластеров
        """
        if self.original_parameters != parameters:
            self.prepare_data(parameters)

        self.calculate_hyperparameters()

        dbscan = DBSCAN(eps=self.optimal_eps, min_samples=self.min_samples)
        self.clusters = dbscan.fit_predict(self.X)
        return self.clusters