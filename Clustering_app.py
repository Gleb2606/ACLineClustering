import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from clustering_methods.DBSCAN import DBSCANClustering
from clustering_methods.Affinity_propagation import AffinityPropagationClustering
from clustering_methods.Agglomerative_clustering import AgglomerativeClustering
from clustering_methods.gaussian_mixture import GaussianMixtureClustering
from clustering_methods.k_means import KMeansClustering
from clustering_methods.mean_shift import MeanShiftClustering

class ClusteringApp:
    """
    Основной класс приложения
    """
    def __init__(self, root):
        """
        Конструктор класса
        :param root:
        """
        self.root = root
        self.root.title("Кластеризация параметров")
        self.root.geometry("1200x800")

        self.clustering_processor = None
        self.checkboxes = []
        self.current_file = None
        self.create_widgets()

    def on_algorithm_select(self, event=None):
        algorithm = self.selected_algorithm.get()
        self.create_params_input(algorithm)
        self.toggle_params_input()

    def create_widgets(self) -> None:
        """
        Метод создания виджетов
        """
        # Выбор файла
        self.file_frame = ttk.LabelFrame(self.root, text="Выбор файла")
        self.file_frame.pack(pady=10, padx=10, fill="x")

        self.btn_browse = ttk.Button(self.file_frame, text="Загрузить файл", command=self.load_file)
        self.btn_browse.pack(side=tk.LEFT, padx=5)

        # Выбор параметров
        self.param_frame = ttk.LabelFrame(self.root, text="Выбор параметров для кластеризации")
        self.param_frame.pack(pady=10, padx=10, fill="x")
        self.canvas = tk.Canvas(self.param_frame)
        self.scrollbar = ttk.Scrollbar(self.param_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Выбор метода кластеризации
        self.algo_frame = ttk.Labelframe(self.root, text="Выбор алгоритма")
        self.algo_frame.pack(pady=10, padx=10, fill="x")

        self.selected_algorithm = tk.StringVar()
        algorithms = [
            ("DBSCAN", "DBSCAN"),
            ("Agglomerative", "AgglomerativeClustering"),
            ("k-Means", "KMeans"),
            ("Mean Shift", "MeanShift"),
            ("Gaussian", "GaussianMixture"),
            ("Affinity Propagation", "AffinityPropagation")
        ]
        self.selected_algorithm.trace_add('write', lambda *_: self.on_algorithm_select())

        for text, value in  algorithms:
            rb = ttk.Radiobutton(
                self.algo_frame,
                text=text,
                variable=self.selected_algorithm,
                value=value
            )
            rb.pack(side=tk.LEFT, padx=5)

        # Подбор гиперпараметров
        self.params_frame = ttk.LabelFrame(self.root, text="Значения гиперпараметров")
        self.params_frame.pack(pady=10, padx=10, fill="x")

        # Переключатель для автоматического подбора гиперпараметров
        self.auto_params_var = tk.BooleanVar(value=True)
        self.autoCheck = ttk.Checkbutton(
            self.params_frame,
            text="Автоматический подбор гиперпараметров",
            variable=self.auto_params_var,
            command=lambda: [self.toggle_params_input()]
        )
        self.autoCheck.pack(anchor='w', padx=5, pady=2)
        self.autoCheck.pack(anchor='w', padx=5, pady=2)
        self.autoCheck.configure(command=self.toggle_params_input)

        # Контейнер для полей ввода
        self.params_input_frame = ttk.Frame(self.params_frame)
        self.params_input_frame.pack(fill="x", padx=5, pady=5)

        # Выполнение кластеризации
        self.ctrl_frame = ttk.Frame(self.root)
        self.ctrl_frame.pack(pady=10, padx=10, fill="x")

        self.btn_cluster = ttk.Button(self.ctrl_frame, text="Выполнить кластеризацию", command=self.run_clustering)
        self.btn_cluster.pack(side=tk.LEFT, padx=5)

        # Сохранение результата
        self.btn_save = ttk.Button(self.ctrl_frame, text="Сохранить результат", command=self.save_results)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        # Отображение результатов
        self.results_frame = ttk.Frame(self.root)
        self.results_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # Данные по кластерам
        self.stats_text = scrolledtext.ScrolledText(self.results_frame, height=10, wrap=tk.WORD)
        self.stats_text.pack(side=tk.LEFT, fill="both", expand=True, padx=5)

        # График
        self.plot_frame = ttk.Frame(self.results_frame)
        self.plot_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5)

    def load_file(self) -> None:
        """
        Метод загрузки файла
        """
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.current_file = file_path
                with open(file_path, 'r') as f:
                    header = f.readline().strip().split(';')

                # Очистка предыдущих чекбоксов
                for cb in self.checkboxes:
                    cb.destroy()
                self.checkboxes = []

                # Создание новых чекбоксов
                for col in header:
                    var = tk.BooleanVar()
                    cb = ttk.Checkbutton(
                        self.scrollable_frame,
                        text=col,
                        variable=var,
                        onvalue=True,
                        offvalue=False
                    )
                    cb.var = var
                    cb.pack(anchor='w', padx=5, pady=2)
                    self.checkboxes.append(cb)

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при чтении файла: {str(e)}")

    def get_selected_parameters(self) -> list:
        """
        Метод получения выбранных параметров
        :return: Массив выбранных параметров
        """
        return [cb['text'] for cb in self.checkboxes if cb.var.get()]

    def toggle_params_input(self) -> None:
        """
        Метод переключения состояния для полей ввода
        """
        state = 'disabled' if self.auto_params_var.get() else 'normal'
        # Итерируемся только по полям ввода внутри params_input_frame
        for child in self.params_input_frame.winfo_children():
            if isinstance(child, ttk.Entry):
                child.configure(state=state)

    def create_params_input(self, algorithm: str) -> None:
        """
        Метод создания полей для выбранного алгоритма
        :param algorithm: Выбранный алгоритм кластеризации
        """
        # Очистка предыдущих полей
        for widget in self.params_input_frame.winfo_children():
            widget.destroy()

        processor_classes = {
            "DBSCAN": DBSCANClustering,
            "AffinityPropagation": AffinityPropagationClustering,
            "KMeans": KMeansClustering,
            "AgglomerativeClustering": AgglomerativeClustering,
            "GaussianMixture": GaussianMixtureClustering,
            "MeanShift": MeanShiftClustering
        }

        if algorithm not in processor_classes:
            return

        try:
            # Создаем временный экземпляр для получения параметров
            processor = processor_classes[algorithm]("dummy_path")
            default_params = processor.get_default_params()

            # Создание полей ввода
            row = 0
            for param, value in default_params.items():
                label = ttk.Label(self.params_input_frame, text=f"{param}:")
                label.grid(row=row, column=0, padx=5, pady=2, sticky='w')

                entry = ttk.Entry(self.params_input_frame)
                entry.insert(0, str(value))
                entry.grid(row=row, column=1, padx=5, pady=2, sticky='ew')

                if self.auto_params_var.get():
                    entry.configure(state='disabled')

                row += 1

            self.params_input_frame.columnconfigure(1, weight=1)

        except Exception as e:
            print(f"Ошибка при создании полей ввода: {str(e)}")

    def get_user_params(self) -> dict:
        """
        Метод возвращающий введенные пользователем параметры
        :return: Словарь со введенными параметрами
        """
        params = {}
        for child in self.params_input_frame.winfo_children():
            if isinstance(child, ttk.Entry):
                label_widget = self.params_input_frame.grid_slaves(
                    row=child.grid_info()["row"],
                    column=0
                )[0]
                param_name = label_widget.cget("text").replace(':', '')

                # Преобразуем значение
                try:
                    value = float(child.get())
                    params[param_name] = int(value) if value.is_integer() else value
                except ValueError:
                    params[param_name] = child.get()
        return params

    def run_clustering(self) -> None:
        """
        Метод выполнения кластеризации
        """
        if not self.current_file:
            messagebox.showerror("Ошибка", "Не загружен файл для кластеризации")
            return

        algorithm = self.selected_algorithm.get()
        parameters = self.get_selected_parameters()

        if not algorithm:
            messagebox.showwarning("Ошибка", "Не выбран алгоритм кластеризации")
            return

        if len(parameters) < 2:
            messagebox.showerror(
                "Ошибка",
                "Выберите от двух параметров для кластеризации"
            )
            return

        try:
            # Создание соответствующего обработчика
            processor_classes = {
                "DBSCAN": DBSCANClustering,
                "AffinityPropagation": AffinityPropagationClustering,
                "KMeans": KMeansClustering,
                "AgglomerativeClustering": AgglomerativeClustering,
                "GaussianMixture": GaussianMixtureClustering,
                "MeanShift": MeanShiftClustering
            }

            # Получение параметров
            auto_params = self.auto_params_var.get()
            user_params = self.get_user_params() if not auto_params else {}

            # Инициализация обработчика
            self.clustering_processor = processor_classes[algorithm](self.current_file)
            self.clustering_processor.auto_params = auto_params

            # Выполнение кластеризации
            self.clustering_processor.perform_clustering(parameters, user_params)

            # Обновление интерфейса
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, self.clustering_processor.get_statistics())
            self.update_plot()

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def update_plot(self) -> None:
        """
        Метод построения графиков
        """
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        plot_data = self.clustering_processor.get_plot_data()
        fig = plt.figure(figsize=(6, 4), dpi=100)

        if len(plot_data['parameters']) == 2:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(
                plot_data['df_clean'][f"{plot_data['parameters'][0]}_clean"],
                plot_data['df_clean'][f"{plot_data['parameters'][1]}_clean"],
                c=plot_data['clusters'],
                cmap='viridis',
                alpha=0.6
            )
            ax.set_xlabel(plot_data['parameters'][0])
            ax.set_ylabel(plot_data['parameters'][1])
            ax.set_title('Кластеризация')
            fig.colorbar(scatter, ax=ax, label='Clusters')
            ax.grid(True)

        elif len(plot_data['parameters']) == 3:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                plot_data['df_clean'][f"{plot_data['parameters'][0]}_clean"],
                plot_data['df_clean'][f"{plot_data['parameters'][1]}_clean"],
                plot_data['df_clean'][f"{plot_data['parameters'][2]}_clean"],
                c=plot_data['clusters'],
                cmap='viridis'
            )
            ax.set_xlabel(plot_data['parameters'][0])
            ax.set_ylabel(plot_data['parameters'][1])
            ax.set_zlabel(plot_data['parameters'][2])
            ax.set_title('Кластеризация')

        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_results(self) -> None:
        """
        Метод сохранения результатов
        """
        if not self.clustering_processor or self.clustering_processor.clusters is None:
            messagebox.showwarning("Warning", "Сначала выполните кластеризацию!")
            return

        try:
            # Получаем данные для сохранения
            save_df = self.clustering_processor.get_cluster_data()

            # Диалог сохранения файла
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Сохранить результаты"
            )

            if file_path:
                save_df.to_csv(file_path, sep=';', index=False)
                messagebox.showinfo("Успех", "Файл успешно сохранен!")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ClusteringApp(root)
    root.mainloop()