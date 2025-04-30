# Импорт необходимых библиотек
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from data_preprocess.scale_data import *
from data_preprocess.plot_builder import *
from data_preprocess.show_statistics import *

# Данные для кластеризации
parameters = ['PSRType(A)', 'r']
file_path = "C:\\Users\\Umaro\\PycharmProjects\\ACLineClustering\\clustering_files\\ACLineSegment_2025_03_28.csv"

# Загрузка данных
X, df_clean = data_scale(file_path, parameters, Float.FLOAT64, 0)

# Автоматический подбор гиперпараметров
min_samples = len(parameters) * X.shape[1]
neigh = NearestNeighbors(n_neighbors=min_samples)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
k_distances = np.sort(distances[:, min_samples-1])
differences = k_distances[1:] - k_distances[:-1]
max_diff_idx = np.argmax(differences)
optimal_eps = k_distances[max_diff_idx]

# Кластеризация DBSCAN
dbscan_data = DBSCAN(eps=optimal_eps, min_samples=min_samples)
clusters = dbscan_data.fit_predict(X)

# Визуализация
plot(df_clean, parameters, clusters, "DBSCAN")

# Вывод статистики по кластерам
show_statistics(df_clean, parameters, clusters)