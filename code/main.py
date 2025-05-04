import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
import imutils
import time
from collections import Counter
import playsound
import threading
import os

class FPSCounter:
    def __init__(self):
        self._start_time = None
        self._frame_count = 0
        self._current_fps = 0
        self._last_update = 0
        
    def start(self):
        self._start_time = time.time()
        self._last_update = self._start_time
        return self
        
    def update(self):
        self._frame_count += 1
        now = time.time()
        if now - self._last_update >= 1.0:
            self._current_fps = self._frame_count / (now - self._start_time)
            self._last_update = now
            self._frame_count = 0
            self._start_time = now
            
    def fps(self):
        return self._current_fps

class AnimalDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detección de Animales")
        self.root.state('zoomed')
        
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
                    "car", "cat", "chair", "cow", "dining-table", "dog", "horse", "motorbike", 
                    "person", "potted plant", "sheep", "sofa", "train", "monitor"]
        self.REQ_CLASSES = ["bird", "cat", "cow", "dog", "horse", "sheep"]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        
        # Variables de control
        self.detecting = False
        self.video_path = ""
        self.cap = None
        self.net = None
        self.fps = None
        self.conf_thresh = 0.2
        self.alarm_active = False
        
        # Nuevas variables para el informe
        self.detection_log = []
        self.current_detections = {}  # Rastrea animales en tiempo real
        
        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo (controles)
        self.control_panel = ttk.LabelFrame(self.main_frame, text="Controles", padding=(10, 5))
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Panel central (video)
        self.video_panel = ttk.LabelFrame(self.main_frame, text="Vista previa", padding=(10, 5))
        self.video_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_label = ttk.Label(self.video_panel)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Panel derecho (registro y estado)
        self.status_panel = ttk.LabelFrame(self.main_frame, text="Estado y Registros", padding=(10, 5))
        self.status_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        
        # Controles
        self.create_controls()
        self.create_event_log()
        self.create_status_bar()

    def create_controls(self):
        ttk.Label(self.control_panel, text="Fuente de video:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.source_var = tk.StringVar(value="Cámara")
        ttk.Radiobutton(self.control_panel, text="Cámara", variable=self.source_var, value="Cámara").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(self.control_panel, text="Archivo", variable=self.source_var, value="Archivo").grid(row=2, column=0, sticky=tk.W)
        
        self.file_entry = ttk.Entry(self.control_panel, state='disabled')
        self.file_entry.grid(row=3, column=0, sticky=tk.EW, pady=(0, 10))
        
        self.browse_btn = ttk.Button(self.control_panel, text="Examinar...", command=self.browse_file, state='disabled')
        self.browse_btn.grid(row=4, column=0, sticky=tk.EW, pady=(0, 20))
        
        ttk.Label(self.control_panel, text="Umbral de confianza:").grid(row=6, column=0, sticky=tk.W)
        self.threshold_slider = ttk.Scale(self.control_panel, from_=0.1, to=0.9, value=0.2, 
                                         command=lambda v: self.threshold_var.set(f"{float(v):.2f}"))
        self.threshold_slider.grid(row=7, column=0, sticky=tk.EW)
        self.threshold_var = tk.StringVar(value="0.20")
        ttk.Label(self.control_panel, textvariable=self.threshold_var).grid(row=8, column=0, sticky=tk.E)
        
        self.start_btn = ttk.Button(self.control_panel, text="Iniciar Detección", command=self.start_detection)
        self.start_btn.grid(row=9, column=0, pady=(20, 5), sticky=tk.EW)
        
        self.stop_btn = ttk.Button(self.control_panel, text="Detener", command=self.stop_detection, state='disabled')
        self.stop_btn.grid(row=10, column=0, pady=(0, 20), sticky=tk.EW)
        
        ttk.Separator(self.control_panel).grid(row=11, column=0, pady=10, sticky="ew")
    
        # Frame para checkboxes
        self.animal_filter_frame = ttk.LabelFrame(self.control_panel, text="Filtrar informe por:", padding=(10, 5))
        self.animal_filter_frame.grid(row=12, column=0, sticky="ew", pady=(0, 10))
        
        # Variables para los checkboxes (1 por animal)
        self.animal_vars = {
            animal: tk.BooleanVar(value=True)  # Todos activos por defecto
            for animal in self.REQ_CLASSES
        }
        
        # Crear checkboxes dinámicamente
        for idx, animal in enumerate(self.REQ_CLASSES):
            cb = ttk.Checkbutton(
                self.animal_filter_frame,
                text=animal.capitalize(),
                variable=self.animal_vars[animal],
                onvalue=True,
                offvalue=False
            )
            cb.grid(row=idx // 2, column=idx % 2, sticky="w", padx=5, pady=2)

        # Botón de exportación
        self.export_btn = ttk.Button(self.control_panel, text="Exportar Informe", command=self.export_report)
        self.export_btn.grid(row=11, column=0, pady=(10, 0), sticky=tk.EW)
        
        self.control_panel.columnconfigure(0, weight=1)
        self.source_var.trace_add('write', self.handle_source_change)

    def create_event_log(self):
        self.event_log = scrolledtext.ScrolledText(self.status_panel, width=40, height=15, state='disabled')
        self.event_log.pack(fill=tk.BOTH, expand=True)
        
        self.detection_frame = ttk.Frame(self.status_panel)
        self.detection_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(self.detection_frame, text="Detecciones recientes:").pack(anchor=tk.W)
        
        self.detection_counters = {}
        for animal in self.REQ_CLASSES:
            frame = ttk.Frame(self.detection_frame)
            frame.pack(fill=tk.X, pady=(2, 0))
            ttk.Label(frame, text=f"{animal.capitalize()}:").pack(side=tk.LEFT)
            var = tk.StringVar(value="0")
            self.detection_counters[animal] = var
            ttk.Label(frame, textvariable=var).pack(side=tk.RIGHT)

    def create_status_bar(self):
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.status_var = tk.StringVar(value="Listo")
        ttk.Label(self.status_bar, textvariable=self.status_var).pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="FPS: 0.00")
        ttk.Label(self.status_bar, textvariable=self.fps_var).pack(side=tk.RIGHT)

    def handle_source_change(self, *args):

        self.detection_log = []
        self.current_detections = {}

        if self.source_var.get() == "Archivo":
            self.file_entry.config(state='normal')
            self.browse_btn.config(state='normal')
        else:
            self.file_entry.config(state='disabled')
            self.browse_btn.config(state='disabled')
            self.file_entry.delete(0, tk.END)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def load_model(self):
        try:
            proto = "C:/Users/Angel/Desktop/files/models/MobileNetSSD_deploy.prototxt.txt"
            model = "C:/Users/Angel/Desktop/files/models/MobileNetSSD_deploy.caffemodel"
            
            if not os.path.exists(proto):
                raise FileNotFoundError(f"No se encontró el archivo prototxt: {proto}")
            if not os.path.exists(model):
                raise FileNotFoundError(f"No se encontró el modelo: {model}")
            
            self.net = cv2.dnn.readNetFromCaffe(proto, model)
            self.log_event("Modelo cargado correctamente")
        except Exception as e:
            self.log_event(f"Error al cargar el modelo: {str(e)}", "error")
            messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{str(e)}")

    def log_event(self, message, level="info"):
        self.event_log.config(state='normal')
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.event_log.insert(tk.END, f"[{timestamp}] {message}\n")
        
        if level == "error":
            self.event_log.tag_add("error", "end-1c linestart", "end-1c lineend")
            self.event_log.tag_config("error", foreground="red")
        elif level == "warning":
            self.event_log.tag_add("warning", "end-1c linestart", "end-1c lineend")
            self.event_log.tag_config("warning", foreground="orange")
        
        self.event_log.see(tk.END)
        self.event_log.config(state='disabled')

    def start_detection(self):
        if self.detecting:
            return
            
        self.detection_log = []  # <-- Vacía la lista al iniciar
        self.current_detections = {}  # <-- También resetea los temporizadores

        if self.source_var.get() == "Cámara":
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.log_event("No se pudo acceder a la cámara", "error")
                messagebox.showerror("Error", "No se pudo acceder a la cámara")
                return
            self.log_event("Usando cámara como fuente de video")
        else:
            video_path = self.file_entry.get()
            if not video_path:
                messagebox.showwarning("Advertencia", "Seleccione un archivo de video primero")
                return
                
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                self.log_event(f"No se pudo abrir el archivo de video: {video_path}", "error")
                messagebox.showerror("Error", f"No se pudo abrir el archivo de video:\n{video_path}")
                return
            self.log_event(f"Reproduciendo video: {os.path.basename(video_path)}")

        for animal in self.REQ_CLASSES:
            self.detection_counters[animal].set("0")
        
        self.detecting = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_var.set("Detectando animales...")
        
        self.fps = FPSCounter().start()
        self.detection_thread = threading.Thread(target=self.detect_animals, daemon=True)
        self.detection_thread.start()

    def stop_detection(self):
        if not self.detecting:
            return
            
        self.detecting = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set("Detección detenida")
        self.log_event("Detección detenida por el usuario")

    def detect_animals(self):
        detection_history = []
        frame_count = 0
        last_alert_time = 0
        alert_cooldown = 5
        
        try:
            while self.detecting and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    self.log_event("Fin del video alcanzado", "warning")
                    break
                
                frame = imutils.resize(frame, width=800)
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                
                self.net.setInput(blob)
                detections = self.net.forward()
                
                detections_in_frame = []
                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > float(self.threshold_var.get()):
                        idx = int(detections[0, 0, i, 1])
                        if self.CLASSES[idx] in self.REQ_CLASSES:
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (36, 255, 12), 2)
                            label = f"{self.CLASSES[idx]}: {confidence * 100:.1f}%"
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                            
                            animal = self.CLASSES[idx]
                            detections_in_frame.append(animal)
                            
                            # Registro de tiempos
                            if animal not in self.current_detections:
                                self.current_detections[animal] = {
                                    "start_time": time.time(),
                                    "end_time": None
                                }
                
                # Actualizar contadores y detecciones que terminaron
                for animal in set(detections_in_frame):
                    current = int(self.detection_counters[animal].get())
                    self.detection_counters[animal].set(str(current + 1))
                
                for animal in list(self.current_detections.keys()):
                    if animal not in detections_in_frame:
                        self.current_detections[animal]["end_time"] = time.time()
                        duration = self.current_detections[animal]["end_time"] - self.current_detections[animal]["start_time"]
                        self.detection_log.append({
                            "animal": animal,
                            "hora": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "duracion": f"{duration:.2f} segundos"
                        })
                        del self.current_detections[animal]
                
                detection_history.append(1 if detections_in_frame else 0)
                if len(detection_history) > 36:
                    detection_history.pop(0)
                    
                    current_time = time.time()
                    if sum(detection_history) > 15 and not self.alarm_active and (current_time - last_alert_time) > alert_cooldown:
                        self.trigger_alarm()
                        last_alert_time = current_time
                
                self.update_video_display(frame)
                self.fps.update()
                frame_count += 1
                if frame_count % 5 == 0:
                    self.fps_var.set(f"FPS: {self.fps.fps():.2f}")
                
                time.sleep(0.01)
            
        except Exception as e:
            self.log_event(f"Error durante la detección: {str(e)}", "error")
            messagebox.showerror("Error", f"Ocurrió un error durante la detección:\n{str(e)}")
        
        finally:
            if hasattr(self, 'fps'):
                self.fps.stop()
                
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            
            cv2.destroyAllWindows()
            self.detecting = False
            self.alarm_active = False
            self.root.after(0, self.update_ui_after_stop)

    def update_ui_after_stop(self):
        self.stop_btn.config(state='disabled')
        self.start_btn.config(state='normal')
        self.status_var.set("Listo")
        self.log_event(f"Procesamiento completado. FPS promedio: {self.fps.fps():.2f}")

    def update_video_display(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def trigger_alarm(self):
        self.alarm_active = True
        self.log_event("¡Intrusión de animal detectada!", "warning")
        
        def play_alarm():
            try:
                playsound.playsound("alert/ringtone.mp3", block=False)
            except Exception as e:
                self.log_event(f"No se pudo reproducir la alarma: {str(e)}", "error")
            finally:
                self.alarm_active = False
        
        threading.Thread(target=play_alarm, daemon=True).start()

    def export_report(self):
        if not self.detection_log:
            messagebox.showwarning("Advertencia", "No hay datos para exportar.")
            return

                # Filtrar detecciones según los checkboxes activos
        filtered_detections = [
            entry for entry in self.detection_log
            if self.animal_vars[entry["animal"]].get()  # Solo animales seleccionados
        ]
        
        if not filtered_detections:
            messagebox.showwarning("Advertencia", "No hay datos con los filtros actuales.")
            return
        
        # Generar informe solo con los animales seleccionados
        report_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Archivo de texto", "*.txt")],
            title="Guardar informe filtrado"
        )
        
        if report_path:
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write("=== INFORME FILTRADO ===\n")
                    f.write(f"Animales incluidos: {', '.join([a for a in self.REQ_CLASSES if self.animal_vars[a].get()])}\n\n")
                    
                    for entry in filtered_detections:
                        f.write(f"Animal: {entry['animal']}\n")
                        f.write(f"Hora: {entry['hora']}\n")
                        f.write(f"Duración: {entry['duracion']}\n")
                        f.write("-" * 30 + "\n")
                
                self.log_event(f"Informe filtrado exportado: {report_path}")
                messagebox.showinfo("Éxito", f"Informe guardado en:\n{report_path}")
            except Exception as e:
                self.log_event(f"Error al exportar: {str(e)}", "error")
                messagebox.showerror("Error", f"No se pudo guardar el informe:\n{str(e)}")

            report_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Archivo de texto", "*.txt"), ("Todos los archivos", "*.*")],
                title="Guardar informe como"
            )
        
        if report_path:
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write("=== INFORME DE DETECCIONES DE ANIMALES ===\n\n")
                    for entry in self.detection_log:
                        f.write(f"Animal: {entry['animal']}\n")
                        f.write(f"Hora: {entry['hora']}\n")
                        f.write(f"Duración: {entry['duracion']}\n")
                        f.write("-" * 30 + "\n")
                
                self.log_event(f"Informe exportado: {report_path}")
                messagebox.showinfo("Éxito", f"Informe guardado en:\n{report_path}")
            except Exception as e:
                self.log_event(f"Error al exportar: {str(e)}", "error")
                messagebox.showerror("Error", f"No se pudo guardar el informe:\n{str(e)}")

    def on_closing(self):
        if self.detecting:
            self.stop_detection()
            time.sleep(0.5)
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    app = AnimalDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
