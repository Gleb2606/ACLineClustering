# Импорт необходимых библиотек
from sklearn.mixture import GaussianMixture
from data_preprocess.scale_data import *
from data_preprocess.plot_builder import *
from data_preprocess.show_statistics import *

# Данные для кластеризации
first_parameter = 'x'
second_parameter = 'r'
file_path = "C:\\Users\\Umaro\\PycharmProjects\\ACLineClustering\\clustering_files\\ACLineSegment_2025_03_28.csv"

# Загрузка данных
X, df_clean = data_scale(file_path, first_parameter, second_parameter, Float.FLOAT64, 0)

# Подбор оптимального количества кластеров через BIC
bic = []
n_components_range = range(1, 11)
for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X)
    bic.append(gmm.bic(X))

# Выбор оптимального числа кластеров
optimal_n = n_components_range[np.argmin(bic)]

# Кластеризация GaussianMixture
gmm = GaussianMixture(n_components=optimal_n, random_state=42)
clusters = gmm.fit_predict(X)

# Визуализация
plot(df_clean, first_parameter, second_parameter, clusters, 'Gaussian Mixture')

# Вывод статистики по кластерам
show_statistics(df_clean, first_parameter, second_parameter, clusters)
