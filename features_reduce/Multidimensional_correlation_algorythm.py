import pandas as pd
from itertools import combinations


def find_correlation_groups_in_range(file_path, lower_bound, upper_bound):
    """
    Находит группы признаков, где все попарные корреляции находятся в заданном диапазоне [lower_bound, upper_bound]

    Parameters:
    file_path (str): путь к CSV-файлу с данными
    lower_bound (float): нижняя граница диапазона корреляции
    upper_bound (float): верхняя граница диапазона корреляции
    """
    # Загрузка данных
    data = pd.read_csv(file_path, sep=';', decimal=',')
    features = data.columns.tolist()
    n = len(features)

    # Вычисление матрицы корреляций
    corr_matrix = data.corr().values  # Без абсолютных значений

    # Словарь для хранения результатов по размерам групп
    results = {}

    # Шаг 1: Обработка пар (k=2)
    k = 2
    significant_combinations = []
    for i, j in combinations(range(n), 2):
        corr_value = corr_matrix[i, j]
        if lower_bound <= corr_value <= upper_bound:
            comb = frozenset([features[i], features[j]])
            significant_combinations.append(comb)

    if not significant_combinations:
        print("Нет значимых комбинаций в заданном диапазоне.")
        return

    results[k] = significant_combinations

    # Шаг 2: Обработка комбинаций размера k >= 3
    k = 3
    while True:
        prev_combinations = results[k - 1]
        prev_set = set(prev_combinations)

        # Генерация кандидатов размера k
        candidate_combinations = set()
        for i in range(len(prev_combinations)):
            for j in range(i + 1, len(prev_combinations)):
                union_set = prev_combinations[i] | prev_combinations[j]
                if len(union_set) == k:
                    candidate_combinations.add(frozenset(union_set))

        # Проверка подкомбинаций
        significant_combinations = []
        for candidate in candidate_combinations:
            valid = True
            # Проверяем, все ли подмножества размера k-1 являются значимыми
            for subset in combinations(candidate, k - 1):
                if frozenset(subset) not in prev_set:
                    valid = False
                    break

            # Если все подмножества значимы, проверяем все попарные корреляции в кандидате
            if valid:
                # Проверяем все попарные корреляции в группе
                all_in_range = True
                feature_indices = [features.index(f) for f in candidate]

                for i, j in combinations(feature_indices, 2):
                    if not (lower_bound <= corr_matrix[i, j] <= upper_bound):
                        all_in_range = False
                        break

                if all_in_range:
                    significant_combinations.append(candidate)

        if not significant_combinations:
            break

        results[k] = significant_combinations
        k += 1

    # Вывод результатов
    print(f"Группы признаков с корреляцией в диапазоне [{lower_bound}, {upper_bound}]:")
    for size, combs in results.items():
        print(f"\nКомбинации размера {size}:")
        for comb in combs:
            feature_list = list(comb)
            # Выводим также значения корреляций для наглядности
            corr_values = []
            for i, j in combinations([features.index(f) for f in feature_list], 2):
                corr_values.append(f"{features[i]}-{features[j]}: {corr_matrix[i, j]:.3f}")

            print(f"{sorted(feature_list)}")
            print(f"   Корреляции: {', '.join(corr_values)}")


# Пример использования
find_correlation_groups_in_range('единицы_оборудования_(градирни).csv', lower_bound=0.1, upper_bound=0.7)