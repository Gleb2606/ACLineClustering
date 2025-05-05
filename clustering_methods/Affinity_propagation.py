# Импорт необходимых библиотек
from sklearn.cluster import AffinityPropagation
from data_preprocess.scale_data import *
from data_preprocess.plot_builder import *
from data_preprocess.show_statistics import *

class AffinityPropagation:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.original_parameters = []
        self.X = None
        self.df_clean = None
        self.clusters = None
        self.damping = 0.7
        self.random_state = 42

    def prepare_data(self, parameters: list):
        """Подготовка и нормализация данных с очисткой предыдущих результатов"""
        # Сброс предыдущих данных
        self.X = None
        self.df_clean = None
        self.clusters = None

        # Новая обработка данных
        self.X, self.df_clean = data_scale(
            self.file_path,
            parameters,
            Float.FLOAT32,
            0
        )
        self.original_parameters = parameters

    def calculate_hyperparameters(self):
        """Автоматический расчет оптимальных гиперпараметров"""
        # В разработке...

    def perform_clustering(self, parameters: list):
        """Выполнение кластеризации AffinityPropagation"""
        if self.original_parameters != parameters:
            self.prepare_data(parameters)

        self.calculate_hyperparameters()

        affinity = AffinityPropagation(damping=0.7, random_state=42)
        self.clusters = affinity.fit_predict(self.X)
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
