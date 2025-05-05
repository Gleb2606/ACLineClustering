import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from clustering_methods.DBSCAN import DBSCANClustering


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

        # Выполнение кластеризации
        self.ctrl_frame = ttk.Frame(self.root)
        self.ctrl_frame.pack(pady=10, padx=10, fill="x")

        self.btn_cluster = ttk.Button(self.ctrl_frame, text="Выполнить кластеризацию", command=self.run_clustering)
        self.btn_cluster.pack(side=tk.LEFT, padx=5)

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
                    cb.pack(anchor='w', padx=5, pady=2)
                    self.checkboxes.append((col, var))

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при чтении файла: {str(e)}")

    def get_selected_parameters(self) -> list:
        """
        Метод получения выбранных параметров
        :return: Массив выбранных параметров
        """
        return [col for col, var in self.checkboxes if var.get()]

    def run_clustering(self) -> None:
        """
        Метод выполнения кластеризации
        """
        if not self.current_file:
            messagebox.showerror("Ошибка", "Не загружен файл для кластеризации")
            return

        parameters = self.get_selected_parameters()

        if len(parameters) < 2:
            messagebox.showerror(
                "Ошибка",
                "Выберите от двух параметров для кластеризации"
            )
            return

        try:
            # Создаем новый процессор для каждой кластеризации
            self.clustering_processor = DBSCANClustering(self.current_file)
            self.clustering_processor.perform_clustering(parameters)

            # Обновление статистики
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, self.clustering_processor.get_statistics())

            # Обновление графика
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

if __name__ == "__main__":
    root = tk.Tk()
    app = ClusteringApp(root)
    root.mainloop()