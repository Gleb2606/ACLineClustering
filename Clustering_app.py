import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from clustering_methods.DBSCAN import DBSCANClustering


class ClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clustering Tool")
        self.root.geometry("1200x800")

        self.clustering_processor = None
        self.checkboxes = []
        self.current_file = None
        self.create_widgets()

    def create_widgets(self):
        # File Selection
        self.file_frame = ttk.LabelFrame(self.root, text="File Selection")
        self.file_frame.pack(pady=10, padx=10, fill="x")

        self.btn_browse = ttk.Button(self.file_frame, text="Browse CSV", command=self.load_file)
        self.btn_browse.pack(side=tk.LEFT, padx=5)

        # Parameters Selection
        self.param_frame = ttk.LabelFrame(self.root, text="Select Parameters (2-3)")
        self.param_frame.pack(pady=10, padx=10, fill="x")

        # Canvas and Scrollbar
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

        # Clustering Controls
        self.ctrl_frame = ttk.Frame(self.root)
        self.ctrl_frame.pack(pady=10, padx=10, fill="x")

        self.btn_cluster = ttk.Button(self.ctrl_frame, text="Run Clustering", command=self.run_clustering)
        self.btn_cluster.pack(side=tk.LEFT, padx=5)

        # Results Display
        self.results_frame = ttk.Frame(self.root)
        self.results_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # Statistics
        self.stats_text = scrolledtext.ScrolledText(self.results_frame, height=10, wrap=tk.WORD)
        self.stats_text.pack(side=tk.LEFT, fill="both", expand=True, padx=5)

        # Plot
        self.plot_frame = ttk.Frame(self.results_frame)
        self.plot_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5)

    def load_file(self):
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
                messagebox.showerror("Error", f"Failed to read file: {str(e)}")

    def get_selected_parameters(self):
        return [col for col, var in self.checkboxes if var.get()]

    def run_clustering(self):
        if not self.current_file:
            messagebox.showwarning("Warning", "Please select a CSV file first!")
            return

        parameters = self.get_selected_parameters()

        if len(parameters) not in [2, 3]:
            messagebox.showwarning(
                "Invalid Selection",
                "Please select exactly 2 or 3 parameters for clustering!"
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
            messagebox.showerror("Error", str(e))

    def update_plot(self):
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
            ax.set_title('DBSCAN Clustering')
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
            ax.set_title('DBSCAN Clustering')

        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = ClusteringApp(root)
    root.mainloop()