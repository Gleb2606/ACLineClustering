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

    def get_default_params(self) -> dict:
        """
        Метод, возвращающий значения гиперпараметров по умолчанию
        :return: Словарь со значениями гиперпараметров
        """
        return {
            'damping': self.damping,
            'random_state': self.random_state
        }

    def calculate_hyperparameters(self):
        """
        Автоматический расчет оптимальных гиперпараметров
        """
        # В разработке...

    def perform_clustering(self, parameters: list, user_params: dict) -> pd.DataFrame:
        """
        Метод выполнения кластеризации AffinityPropagation
        :param parameters: Список параметров кластеризации
        :param user_params: Словарь с пользовательскими значениями гиперпараметров
        :return: Датафрейм сформированных кластеров
        """
        if self.original_parameters != parameters:
            self.prepare_data(parameters)

        if self.auto_params:
            self.calculate_hyperparameters()
        else:
            self.damping = user_params.get('damping', self.damping)
            self.random_state = user_params.get('random_state', self.random_state)

        affinity = AffinityPropagation(damping=self.damping, random_state=self.random_state)
        self.clusters = affinity.fit_predict(self.X)
        return self.clusters