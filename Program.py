import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mpl_toolkits.mplot3d import Axes3D

class ClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Кластеризация данных")

        # Переменные
        self.file_path = None
        self.headers = []
        self.checkboxes = []
        self.selected_vars = []
        self.clustered_data = None
        self.current_canvas = None

        # Настройка GUI
        self.create_widgets()

    def create_widgets(self):
        # Верхняя панель с кнопками
        control_frame = Frame(self.root)
        control_frame.pack(pady=10, fill=X)

        self.load_btn = Button(control_frame, text="Загрузить CSV", command=self.load_csv)
        self.load_btn.pack(side=LEFT, padx=5)

        # Фрейм для параметров кластеризации
        self.cluster_frame = LabelFrame(self.root, text="Параметры кластеризации")
        self.cluster_frame.pack(padx=10, pady=5, fill=BOTH)

        # Поле для ввода количества кластеров с валидацией
        Label(self.cluster_frame, text="Количество кластеров:").pack(pady=2)
        vcmd = (self.root.register(self.validate_cluster_num), '%P')
        self.cluster_num = Entry(self.cluster_frame, validate='key', validatecommand=vcmd)
        self.cluster_num.pack(pady=2)

        # Кнопки процессов
        self.process_btn = Button(control_frame, text="Выполнить кластеризацию", command=self.process_data)
        self.process_btn.pack(side=LEFT, padx=5)

        self.save_btn = Button(control_frame, text="Сохранить результат", command=self.save_results)
        self.save_btn.pack(side=LEFT, padx=5)

        # Область для графиков
        self.graph_frame = LabelFrame(self.root, text="Визуализация")
        self.graph_frame.pack(padx=10, pady=10, fill=BOTH, expand=True)

    def validate_cluster_num(self, new_value):
        """Валидация ввода для количества кластеров"""
        if new_value == "":
            return True  # Разрешаем пустое поле для временного ввода
        return new_value.isdigit() and int(new_value) > 0

    def load_csv(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            df = pd.read_csv(self.file_path, sep=';', nrows=1)
            self.headers = [h for h in df.columns if h != 'Uid']
            self.create_checkboxes()

    def create_checkboxes(self):
        # Очистка предыдущих чекбоксов
        for cb in self.checkboxes:
            cb.destroy()
        self.selected_vars = []
        self.checkboxes = []

        # Создание новых чекбоксов с прокруткой
        canvas = Canvas(self.cluster_frame)
        scrollbar = ttk.Scrollbar(self.cluster_frame, orient="vertical", command=canvas.yview)
        scroll_frame = Frame(canvas)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        for header in self.headers:
            var = BooleanVar()
            cb = Checkbutton(scroll_frame, text=header, variable=var)
            cb.pack(anchor=W)
            self.checkboxes.append(cb)
            self.selected_vars.append(var)

    def process_data(self):
        # Проверка 1: Файл загружен
        if not self.file_path:
            messagebox.showerror("Ошибка", "Не выбран файл для кластеризации")
            return

        # Проверка 2: Выбрано минимум 2 параметра
        selected = [self.headers[i] for i, var in enumerate(self.selected_vars) if var.get()]
        if len(selected) < 2:
            messagebox.showerror("Ошибка", "Не выбраны параметры для кластеризации (минимум 2)")
            return

        # Проверка 3: Поле количества кластеров заполнено
        cluster_num_text = self.cluster_num.get().strip()
        if not cluster_num_text:
            messagebox.showerror("Ошибка", "Не задано количество кластеров")
            return

        # Проверка 4: Корректное значение количества кластеров
        try:
            n_clusters = int(cluster_num_text)
            if n_clusters <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Количество кластеров должно быть целым положительным числом")
            return

        # Очистка предыдущего графика
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
            self.current_canvas = None

        # Загрузка и обработка данных
        df = pd.read_csv(self.file_path, sep=';')
        label_encoders = {}

        for col in selected:
            original_missing = df[col].isna()
            temp_series = df[col].astype(str).str.replace(',', '.')
            temp_series_numeric = pd.to_numeric(temp_series, errors='coerce')
            new_missing = temp_series_numeric.isna()
            newly_missing = new_missing & ~original_missing

            if newly_missing.any():
                le = LabelEncoder()
                encoded_col = le.fit_transform(temp_series)
                df[col] = encoded_col
                label_encoders[col] = le
            else:
                df[col] = temp_series_numeric

        # Удаление пропусков
        df_clean = df.dropna(subset=selected)
        if df_clean.empty:
            messagebox.showerror("Ошибка", "Нет данных после удаления пропусков")
            return

        # Нормализация и кластеризация
        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean[selected])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_clean['cluster'] = kmeans.fit_predict(X)

        # Сохранение результата
        output_columns = selected + ['cluster']
        self.clustered_data = df_clean[output_columns]

        # Создание графиков
        fig = Figure(figsize=(6, 4), dpi=100)
        if len(selected) == 2:
            ax = fig.add_subplot(111)
            ax.scatter(df_clean[selected[0]], df_clean[selected[1]],
                       c=df_clean['cluster'], cmap='viridis')
            ax.set_xlabel(selected[0])
            ax.set_ylabel(selected[1])
            ax.set_title('Визуализация кластеров')

        elif len(selected) == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df_clean[selected[0]], df_clean[selected[1]], df_clean[selected[2]],
                       c=df_clean['cluster'], cmap='viridis')
            ax.set_xlabel(selected[0])
            ax.set_ylabel(selected[1])
            ax.set_zlabel(selected[2])
            ax.set_title('Визуализация кластеров')

        else:
            return

        # Отображение графика
        self.current_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def save_results(self):
        if self.clustered_data is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Сохранить файл"
            )
            if file_path:
                self.clustered_data.to_csv(file_path, index=False)
                messagebox.showinfo("Успех", "Файл успешно сохранен")
        else:
            messagebox.showerror("Ошибка", "Сначала выполните кластеризацию")

if __name__ == "__main__":
    root = Tk()
    app = ClusteringApp(root)
    root.mainloop()