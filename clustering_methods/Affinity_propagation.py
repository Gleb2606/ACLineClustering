# Импорт необходимых библиотек
import pandas as pd
from sklearn.cluster import AffinityPropagation
from clustering_methods.Base_clustering import BaseClustering

class AffinityPropagationClustering(BaseClustering):
    """
    Класс кластеризации методом Affinity propagation
    """
    def __init__(self, file_path: str):
        """
        Конструктор класса
        :param file_path: Путь к файлу
        """
        super().__init__(file_path)
        self.damping = 0.7
        self.random_state = 42

    def calculate_hyperparameters(self):
        """Автоматический расчет оптимальных гиперпараметров"""
        # В разработке...

    def perform_clustering(self, parameters: list) -> pd.DataFrame:
        """
        Метод выполнения кластеризации AffinityPropagation
        :param parameters: Список параметров кластеризации
        :return: Датафрейм сформированных кластеров
        """
        if self.original_parameters != parameters:
            self.prepare_data(parameters)

        self.calculate_hyperparameters()

        affinity = AffinityPropagation(damping=0.7, random_state=42)
        self.clusters = affinity.fit_predict(self.X)
        return self.clusters