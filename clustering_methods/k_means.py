# Импорт необходимых библиотек
import pandas as pd
from sklearn.cluster import KMeans
from clustering_methods.Base_clustering import BaseClustering

class KMeansClustering(BaseClustering):
    """
    Класс кластеризации методом К-средних
    """
    def __init__(self, file_path: str):
        """
        Конструктор класса
        :param file_path: Путь к файлу
        """
        super().__init__(file_path)
        self.n_clusters = 2

    def get_default_params(self) -> dict:
        """
        Метод, возвращающий значения гиперпараметров по умолчанию
        :return: Словарь со значениями гиперпараметров
        """
        return {
            'n_clusters': self.n_clusters
        }

    def calculate_hyperparameters(self) -> None:
        """
        Автоматический расчет оптимальных гиперпараметров
        """
        # В разработке...

    def perform_clustering(self, parameters: list, user_params: dict) -> pd.DataFrame:
        """
        Выполнение кластеризации K-means
        :param parameters: Список параметров кластеризации
        :param user_params: Словарь с пользовательскими значениями гиперпараметров
        :return: Датафрейм сформированных кластеров
        """
        if self.original_parameters != parameters:
            self.prepare_data(parameters)

        if self.auto_params:
            self.calculate_hyperparameters()
        else:
            self.n_clusters = user_params.get('n_clusters', self.n_clusters)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.clusters = kmeans.fit_predict(self.X)
        return self.clusters