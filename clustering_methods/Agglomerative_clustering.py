# Импорт необходимых библиотек
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from data_preprocess.scale_data import *
from data_preprocess.plot_builder import *
from data_preprocess.show_statistics import *

# Данные для кластеризации
parameters = ['r', 'x']
file_path = "C:\\Users\\Umaro\\PycharmProjects\\ACLineClustering\\clustering_files\\ACLineSegment_2025_03_28.csv"

# Загрузка данных
X, df_clean = data_scale(file_path, parameters, Float.FLOAT64, 0)

# Построение иерархии методом Уорда
Z = linkage(X, method='ward')
dendrogram_builder(Z)

# Подбор числа кластеров
distances = Z[:, 2]
diff = np.diff(distances)
optimal_num_clusters = np.argmax(diff[::-1]) + 2

# Кластеризация AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=optimal_num_clusters, linkage='ward')
clusters = agg.fit_predict(X)

# Визуализация
plot(df_clean, parameters, clusters, "Agglomerative Clustering")

# Вывод статистики по кластерам
show_statistics(df_clean, parameters, clusters)