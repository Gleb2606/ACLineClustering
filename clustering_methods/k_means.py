# Импорт необходимых библиотек
from sklearn.cluster import KMeans
from data_preprocess.scale_data import *
from data_preprocess.plot_builder import *
from data_preprocess.show_statistics import *

# Данные для кластеризации
first_parameter = 'x'
second_parameter = 'r'
file_path = "C:\\Users\\Umaro\\PycharmProjects\\ACLineClustering\\clustering_files\\ACLineSegment_2025_03_28.csv"

# Загрузка данных
X, df_clean = data_scale(file_path, first_parameter, second_parameter, Float.FLOAT64, 0)

# Кластеризация K-means
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# Визуализация
plot(df_clean, first_parameter, second_parameter, clusters, 'k-means')

# Вывод статистики по кластерам
show_statistics(df_clean, first_parameter, second_parameter, clusters)