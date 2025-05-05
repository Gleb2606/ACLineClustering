# Импорт необходимых библиотек
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from data_preprocess.scale_data import *
from data_preprocess.show_statistics import *

class AgglomerativeClustering:
    """
    Класс кластеризации методом иерархической кластеризации
    """
    def __init__(self, file_path: str):
        """
        Конструктор класса
        :param file_path: Путь к файлу
        """
        self.file_path = file_path
        self.original_parameters = []
        self.X = None
        self.df_clean = None
        self.clusters = None
        self.optimal_num_clusters = None

    def prepare_data(self, parameters: list) -> None:
        """
        Подготовка и нормализация данных с очисткой предыдущих результатов
        :param parameters: Список параметров кластеризации
        """
        # Сброс предыдущих данных
        self.X = None
        self.df_clean = None
        self.clusters = None
        self.optimal_num_clusters = None

        # Новая обработка данных
        self.X, self.df_clean = data_scale(
            self.file_path,
            parameters,
            Float.FLOAT64,
            0
        )
        self.original_parameters = parameters

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

    def get_statistics(self) -> str:
        """
        Метод получения статистики по кластерам
        :return: Статистика по кластерам
        """
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

    def get_plot_data(self) -> dict:
        """
        Метод получения данных для построения графиков
        :return: Данные для построения графиков
        """
        return {
            'df_clean': self.df_clean,
            'parameters': self.original_parameters,
            'clusters': self.clusters
        }

