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
        self.optimal_n = None

    def get_default_params(self) -> dict:
        """
        Метод, возвращающий значения гиперпараметров по умолчанию
        :return: Словарь со значениями гиперпараметров
        """
        return {
            'optimal_n': self.optimal_n
        }

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

    def perform_clustering(self, parameters: list, user_params: dict) -> pd.DataFrame:
        """
        Выполнение кластеризации GaussianMixture
        :param parameters: Список параметров кластеризации
        :param user_params: Словарь с пользовательскими значениями гиперпараметров
        :return: Датафрейм сформированных кластеров
        """
        if self.original_parameters != parameters:
            self.prepare_data(parameters)

        if self.auto_params:
            self.calculate_hyperparameters()
        else:
            self.optimal_n = user_params.get('optimal_n', self.optimal_n)

        gmm = GaussianMixture(n_components=self.optimal_n, random_state=42)
        self.clusters = gmm.fit_predict(self.X)
        return self.clusters