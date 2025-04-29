# Импорт необходимых библиотек
from sklearn.cluster import DBSCAN
from data_preprocess.scale_data import *
from data_preprocess.plot_builder import *
from data_preprocess.show_statistics import *

# Данные для кластеризации
first_parameter = 'x'
second_parameter = 'r'
file_path = "C:\\Users\\Umaro\\PycharmProjects\\ACLineClustering\\clustering_files\\ACLineSegment_2025_03_28.csv"

# Загрузка данных
X, df_clean = data_scale(file_path, first_parameter, second_parameter, Float.FLOAT64, 0)

# Кластеризация DBSCAN
dbscan_data = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan_data.fit_predict(X)

# Визуализация
plot(df_clean, first_parameter, second_parameter, clusters, "DBSCAN")

# Вывод статистики по кластерам
show_statistics(df_clean, first_parameter, second_parameter, clusters)


