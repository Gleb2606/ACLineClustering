# Импорт необходимых библиотек
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from clustering_methods.Base_clustering import BaseClustering

class GaussianMixtureClustering(BaseClustering):
    """
    Класс кластеризации методом Гауссовых смесей
    """
    def __init__(self, file_path: str):
        """
        Конструктор класса
        :param file_path: Путь к файлу
        """
        super().__init__(file_path)
        self.clusters = None
        self.optimal_n = None

    def calculate_hyperparameters(self) -> None:
        """
        Автоматический расчет оптимальных гиперпараметров
        """
        bic = []
        n_components_range = range(1, 11)
        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(self.X)
            bic.append(gmm.bic(self.X))

        # Выбор оптимального числа кластеров
        self.optimal_n = n_components_range[np.argmin(bic)]

    def perform_clustering(self, parameters: list) -> pd.DataFrame:
        """
        Выполнение кластеризации GaussianMixture
        :param parameters: Список параметров кластеризации
        :return: Датафрейм сформированных кластеров
        """
        if self.original_parameters != parameters:
            self.prepare_data(parameters)

        self.calculate_hyperparameters()

        gmm = GaussianMixture(n_components=self.optimal_n, random_state=42)
        self.clusters = gmm.fit_predict(self.X)
        return self.clusters