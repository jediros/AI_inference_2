import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
from ultralytics import YOLO

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Video Processing")
        self.root.geometry("300x400")

        # Cargar el modelo YOLO
        self.model = YOLO('best_class.pt')

        # Botón para seleccionar un video
        self.single_upload_btn = tk.Button(root, text="Select Video", command=self.select_video)
        self.single_upload_btn.pack(pady=5)

        # Botón para seleccionar un directorio
        self.directory_upload_btn = tk.Button(root, text="Select Directory", command=self.select_directory)
        self.directory_upload_btn.pack(pady=5)

        # Campo para umbral de confianza
        self.threshold_label = tk.Label(root, text="Confidence Threshold (0-1):")
        self.threshold_label.pack()

        self.threshold_entry = tk.Entry(root)
        self.threshold_entry.insert(0, "0.3")  # Valor predeterminado
        self.threshold_entry.pack()

        # Botón para procesar un solo video
        self.single_process_btn = tk.Button(root, text="Process Single Video", command=self.process_single_video)
        self.single_process_btn.pack(pady=5)

        # Botón para procesar todos los videos de un directorio
        self.directory_process_btn = tk.Button(root, text="Process Videos in Directory", command=self.process_videos)
        self.directory_process_btn.pack(pady=5)

        # Botón para usar la webcam
        self.webcam_btn = tk.Button(root, text="Use USB Webcam", command=self.use_webcam)
        self.webcam_btn.pack(pady=5)

        # Botón para establecer la resolución de la webcam
        self.resolution_btn = tk.Button(root, text="Set Webcam Resolution", command=self.set_resolution)
        self.resolution_btn.pack(pady=5)

        # Botón para permitir al usuario ingresar una resolución personalizada
        self.custom_resolution_btn = tk.Button(root, text="Set Custom Resolution", command=self.set_custom_resolution)
        self.custom_resolution_btn.pack(pady=5)

        self.video_path = ""
        self.video_directory = ""
        self.confidence_threshold = 0.3
        self.width = 1920  # Valor predeterminado para el ancho
        self.height = 1080  # Valor predeterminado para la altura

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if self.video_path:
            messagebox.showinfo("Selected Video", f"Selected Video: {self.video_path}")

    def select_directory(self):
        self.video_directory = filedialog.askdirectory()
        if self.video_directory:
            messagebox.showinfo("Selected Directory", f"Selected Directory: {self.video_directory}")

    def process_single_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first")
            return
        self._process_video(self.video_path)

    def process_videos(self):
        if not self.video_directory:
            messagebox.showerror("Error", "Please select a directory containing video files")
            return
        video_files = [f for f in os.listdir(self.video_directory) if f.endswith(('.mp4', '.avi'))]
        if not video_files:
            messagebox.showerror("Error", "No video files found in the selected directory")
            return
        for video_file in video_files:
            video_path = os.path.join(self.video_directory, video_file)
            self._process_video(video_path)
        messagebox.showinfo("Done", "Video processing complete. Output saved in the selected directory")

    def _process_video(self, video_path):
        try:
            self.confidence_threshold = float(self.threshold_entry.get())
            if not (0 <= self.confidence_threshold <= 1):
                raise ValueError("Threshold must be between 0 and 1")
        except ValueError:
            messagebox.showerror("Error", "Invalid confidence threshold. It should be a number between 0 and 1.")
            return

        cap = cv2.VideoCapture(video_path)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.splitext(video_path)[0] + '_output.avi', fourcc, 30.0,
                              (int(cap.get(3)), int(cap.get(4))))

        ret = True
        while ret:
            ret, frame = cap.read()
            if ret:
                results = self.model.track(frame, conf=self.confidence_threshold, persist=True)
                frame_ = results[0].plot()

                out.write(frame_)
                cv2.imshow('frame', frame_)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Done", "Video processing complete.")

    def use_webcam(self):
        # Preguntar por el índice de la cámara
        index = simpledialog.askinteger("Camera Index", "Enter the USB camera index (0, 1, 2, etc.):", minvalue=0, maxvalue=10)

        if index is None:
            return  # Cancelado

        # Verificar umbral de confianza
        try:
            self.confidence_threshold = float(self.threshold_entry.get())
            if not (0 <= self.confidence_threshold <= 1):
                raise ValueError("Threshold must be between 0 and 1")
        except ValueError:
            messagebox.showerror("Error", "Invalid confidence threshold. It should be a number between 0 and 1.")
            return

        # Captura desde la webcam
        cap = cv2.VideoCapture(index)

        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open USB webcam")
            return

        # Establecer la resolución de la webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Aplicar el modelo YOLO en el frame
            results = self.model.track(frame, conf=self.confidence_threshold, persist=True)
            frame_ = results[0].plot()

            # Mostrar el frame en vivo con detecciones
            cv2.imshow('Webcam Live', frame_)

            # Salir al presionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def set_resolution(self):
        # Opciones de resolución
        resolutions = {
            "1080p (1920x1080)": (1920, 1080),
            "720p (1280x720)": (1280, 720),
            "480p (640x480)": (640, 480),
            "360p (640x360)": (640, 360)
        }

        # Crear una ventana de selección de resolución
        resolution_window = tk.Toplevel(self.root)
        resolution_window.title("Set Webcam Resolution")

        # Etiqueta para instrucciones
        instructions_label = tk.Label(resolution_window, text="Select a resolution:")
        instructions_label.pack(pady=5)

        # Crear un menú desplegable para seleccionar la resolución
        resolution_var = tk.StringVar(resolution_window)
        resolution_var.set("1080p (1920x1080)")  # Valor predeterminado

        resolution_menu = tk.OptionMenu(resolution_window, resolution_var, *resolutions.keys())
        resolution_menu.pack(pady=5)

        def apply_resolution():
            selected_resolution = resolution_var.get()
            self.width, self.height = resolutions[selected_resolution]
            messagebox.showinfo("Resolution Set", f"Webcam resolution set to {self.width}x{self.height}")
            resolution_window.destroy()

        apply_btn = tk.Button(resolution_window, text="Apply", command=apply_resolution)
        apply_btn.pack(pady=5)

    def set_custom_resolution(self):
        # Solicitar al usuario que ingrese su propia resolución
        resolution_input = simpledialog.askstring("Custom Resolution", "Enter resolution (Width x Height):")
        if resolution_input is None:
            return  # Cancelado

        try:
            width, height = map(int, resolution_input.split('x'))
            self.width, self.height = width, height
            messagebox.showinfo("Resolution Set", f"Custom resolution set to {self.width}x{self.height}")
        except ValueError:
            messagebox.showerror("Error", "Invalid resolution format. Please use 'Width x Height' (e.g., 1920x1080).")

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()
