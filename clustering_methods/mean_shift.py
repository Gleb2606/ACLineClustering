# Импорт необходимых библиотек
import pandas as pd
from sklearn.cluster import MeanShift
from clustering_methods.Base_clustering import BaseClustering

class MeanShiftClustering(BaseClustering):
    """
    Класс кластеризации методом среднего сдвига
    """
    def __init__(self, file_path: str):
        """
        Конструктор класса
        :param file_path: Путь к файлу
        """
        super().__init__(file_path)
        self.bandwidth = 0.8
        self.bin_seeding = True

    def get_default_params(self) -> dict:
        """
        Метод, возвращающий значения гиперпараметров по умолчанию
        :return: Словарь со значениями гиперпараметров
        """
        return {
            'bandwidth': self.bandwidth
        }

    def calculate_hyperparameters(self) -> None:
        """
        Автоматический расчет оптимальных гиперпараметров
        """
        # В разработке...

    def perform_clustering(self, parameters: list, user_params: dict) -> pd.DataFrame:
        """
        Выполнение кластеризации MeanShift
        :param parameters: Список параметров кластеризации
        :param user_params: Словарь с пользовательскими значениями гиперпараметров
        :return: Датафрейм сформированных кластеров
        """
        if self.original_parameters != parameters:
            self.prepare_data(parameters)

        if self.auto_params:
            self.calculate_hyperparameters()
        else:
            self.bandwidth = user_params.get('bandwidth', self.bandwidth)

        ms = MeanShift(bandwidth=self.bandwidth, bin_seeding=self.bin_seeding)
        self.clusters = ms.fit_predict(self.X)
        return self.clusters