import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Датасет
data = "Test_Turbines.csv"

# Загрузка данных
df = pd.read_csv(data, sep=';', decimal=',')

# Расчет матрицы корреляций
corr_matrix = df.corr()

# Настройка визуализации
plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_matrix,
    annot=True,          # Показать значения в ячейках
    fmt=".2f",           # Формат чисел (2 знака после запятой)
    cmap="coolwarm",     # Цветовая схема
    vmin=-1, vmax=1,     # Диапазон значений
    linewidths=0.5,
    square=True          # Квадратные ячейки
)
plt.title("Матрица корреляций", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()