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
        self.optimal_eps = 0.5
        self.min_samples = 5

    def get_default_params(self) -> dict:
        """
        Метод, возвращающий значения гиперпараметров по умолчанию
        :return: Словарь со значениями гиперпараметров
        """
        return {
            'optimal_eps': self.optimal_eps,
            'min_samples': self.min_samples,
        }

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

    def perform_clustering(self, parameters: list, user_params: dict) -> pd.DataFrame:
        """
        Выполнение кластеризации DBSCAN
        :param parameters: Список параметров кластеризации
        :param user_params: Словарь с пользовательскими значениями гиперпараметров
        :return: Датафрейм сформированных кластеров
        """
        if self.original_parameters != parameters:
            self.prepare_data(parameters)

        if self.auto_params:
            self.calculate_hyperparameters()
        else:
            self.optimal_eps = user_params.get('optimal_eps', self.optimal_eps)
            self.min_samples = user_params.get('min_samples', self.min_samples)

        dbscan = DBSCAN(eps=self.optimal_eps, min_samples=self.min_samples)
        self.clusters = dbscan.fit_predict(self.X)
        return self.clusters