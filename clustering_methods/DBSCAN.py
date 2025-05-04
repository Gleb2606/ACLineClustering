# Импорт необходимых библиотек
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from data_preprocess.scale_data import data_scale, Float

class DBSCANClustering:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.original_parameters = []
        self.X = None
        self.df_clean = None
        self.clusters = None
        self.optimal_eps = None
        self.min_samples = None

    def prepare_data(self, parameters: list):
        """Подготовка и нормализация данных с очисткой предыдущих результатов"""
        # Сброс предыдущих данных
        self.X = None
        self.df_clean = None
        self.clusters = None
        self.optimal_eps = None
        self.min_samples = None

        # Новая обработка данных
        self.X, self.df_clean = data_scale(
            self.file_path,
            parameters,
            Float.FLOAT64,
            0
        )
        self.original_parameters = parameters

    def calculate_hyperparameters(self):
        """Автоматический расчет оптимальных гиперпараметров"""
        self.min_samples = len(self.original_parameters) * self.X.shape[1]
        neigh = NearestNeighbors(n_neighbors=self.min_samples)
        nbrs = neigh.fit(self.X)
        distances, _ = nbrs.kneighbors(self.X)
        k_distances = np.sort(distances[:, self.min_samples - 1])
        differences = k_distances[1:] - k_distances[:-1]
        max_diff_idx = np.argmax(differences)
        self.optimal_eps = k_distances[max_diff_idx]

    def perform_clustering(self, parameters: list):
        """Выполнение кластеризации DBSCAN"""
        if self.original_parameters != parameters:
            self.prepare_data(parameters)

        self.calculate_hyperparameters()

        dbscan = DBSCAN(eps=self.optimal_eps, min_samples=self.min_samples)
        self.clusters = dbscan.fit_predict(self.X)
        return self.clusters

    def get_statistics(self):
        """Получение статистики по кластерам"""
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

    def get_plot_data(self):
        """Возвращает данные для построения графиков"""
        return {
            'df_clean': self.df_clean,
            'parameters': self.original_parameters,
            'clusters': self.clusters
        }