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
        self.optimal_num_clusters = None

    def get_default_params(self) -> dict:
        """
        Метод, возвращающий значения гиперпараметров по умолчанию
        :return: Словарь со значениями гиперпараметров
        """
        return {
            'optimal_num_clusters': self.optimal_num_clusters,
        }

    def calculate_hyperparameters(self) -> None:
        """
        Автоматический расчет оптимальных гиперпараметров
        """
        Z = linkage(self.X, method='ward')
        # dendrogram_builder(Z)
        distances = Z[:, 2]
        diff = np.diff(distances)
        self.optimal_num_clusters = np.argmax(diff[::-1]) + 2

    def perform_clustering(self, parameters: list, user_params: dict) -> pd.DataFrame:
        """
        Выполнение кластеризации AgglomerativeClustering
        :param parameters: Список параметров кластеризации
        :param user_params: Словарь с пользовательскими значениями гиперпараметров
        :return: Датафрейм сформированных кластеров
        """
        if self.original_parameters != parameters:
            self.prepare_data(parameters)

        if self.auto_params:
            self.calculate_hyperparameters()
        else:
            self.optimal_num_clusters = user_params.get('optimal_num_clusters', self.optimal_num_clusters)

        ac = AgglomerativeClustering(n_clusters=self.optimal_num_clusters, linkage='ward')
        self.clusters = ac.fit_predict(self.X)
        return self.clusters