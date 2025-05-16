# Импорт необходимых библиотек
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from clustering_methods.Base_clustering import BaseClustering

class AgglomerativeClustering(BaseClustering):
    """
    Класс кластеризации методом иерархической кластеризации
    """
    def __init__(self, file_path: str):
        """
        Конструктор класса
        :param file_path: Путь к файлу
        """
        super().__init__(file_path)
        self.clusters = None
        self.optimal_num_clusters = None

    def calculate_hyperparameters(self) -> None:
        """
        Автоматический расчет оптимальных гиперпараметров
        """
        Z = linkage(self.X, method='ward')
        # dendrogram_builder(Z)
        distances = Z[:, 2]
        diff = np.diff(distances)
        self.optimal_num_clusters = np.argmax(diff[::-1]) + 2

    def perform_clustering(self, parameters: list) -> pd.DataFrame:
        """
        Выполнение кластеризации AgglomerativeClustering
        :param parameters: Список параметров кластеризации
        :return: Датафрейм сформированных кластеров
        """
        if self.original_parameters != parameters:
            self.prepare_data(parameters)

        self.calculate_hyperparameters()

        ac = AgglomerativeClustering(n_clusters=self.optimal_num_clusters, linkage='ward')
        self.clusters = ac.fit_predict(self.X)
        return self.clusters