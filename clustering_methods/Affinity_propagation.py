# Импорт необходимых библиотек
from sklearn.cluster import AffinityPropagation
from data_preprocess.scale_data import *
from data_preprocess.plot_builder import *
from data_preprocess.show_statistics import *

# Данные для кластеризации
first_parameter = 'x'
second_parameter = 'r'
file_path = "C:\\Users\\Umaro\\PycharmProjects\\ACLineClustering\\clustering_files\\ACLineSegment_2025_03_28.csv"

# Загрузка данных
X, df_clean = data_scale(file_path, first_parameter, second_parameter, Float.FLOAT32, 10000)

# Кластеризация методом AffinityPropagation
affinity = AffinityPropagation(damping=0.7, random_state=42)
clusters = affinity.fit_predict(X)

# Визуализация кластеров
plot(df_clean, first_parameter, second_parameter, clusters, "AffinityPropagation")

# Вывод статистики по кластерам
show_statistics(df_clean, first_parameter, second_parameter, clusters)