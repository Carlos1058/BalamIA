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
        self._end_time = None  # Nuevo atributo para el tiempo final
        
    def start(self):
        self._start_time = time.time()
        self._last_update = self._start_time
        return self
        
    def stop(self):  # Nuevo método
        self._end_time = time.time()
        
    def update(self):
        self._frame_count += 1
        now = time.time()
        if now - self._last_update >= 1.0:  # Actualizar FPS cada segundo
            self._current_fps = self._frame_count / (now - self._start_time)
            self._last_update = now
            self._frame_count = 0
            self._start_time = now
            
    def fps(self):
        return self._current_fps
        
    def elapsed(self):
        if self._end_time:  # Si se ha llamado a stop()
            return self._end_time - self._start_time
        elif self._start_time:
            return time.time() - self._start_time
        return 0.0

class AnimalDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detección de Animales")
        self.root.state('zoomed')  # Pantalla completa
        
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
        self.detection_history = []
        self.alarm_active = False
        
        self.setup_ui()
        
        # Cargar modelo
        self.load_model()
        
    def setup_ui(self):
        # Frame principal
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
        
        # Registro de eventos
        self.create_event_log()
        
        # Barra de estado
        self.create_status_bar()
        
    def create_controls(self):
        # Selección de video
        ttk.Label(self.control_panel, text="Fuente de video:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.source_var = tk.StringVar(value="Cámara")
        ttk.Radiobutton(self.control_panel, text="Cámara", variable=self.source_var, value="Cámara").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(self.control_panel, text="Archivo", variable=self.source_var, value="Archivo").grid(row=2, column=0, sticky=tk.W)
        
        self.file_entry = ttk.Entry(self.control_panel, state='disabled')
        self.file_entry.grid(row=3, column=0, sticky=tk.EW, pady=(0, 10))
        
        self.browse_btn = ttk.Button(self.control_panel, text="Examinar...", command=self.browse_file, state='disabled')
        self.browse_btn.grid(row=4, column=0, sticky=tk.EW, pady=(0, 20))
        
        # Configuración de detección
        ttk.Label(self.control_panel, text="Configuración:").grid(row=5, column=0, sticky=tk.W, pady=(0, 5))
        
        ttk.Label(self.control_panel, text="Umbral de confianza:").grid(row=6, column=0, sticky=tk.W)
        self.threshold_slider = ttk.Scale(self.control_panel, from_=0.1, to=0.9, value=0.2, 
                                         command=lambda v: self.threshold_var.set(f"{float(v):.2f}"))
        self.threshold_slider.grid(row=7, column=0, sticky=tk.EW)
        
        self.threshold_var = tk.StringVar(value="0.20")
        ttk.Label(self.control_panel, textvariable=self.threshold_var).grid(row=8, column=0, sticky=tk.E)
        
        # Botones de control
        self.start_btn = ttk.Button(self.control_panel, text="Iniciar Detección", command=self.start_detection)
        self.start_btn.grid(row=9, column=0, pady=(20, 5), sticky=tk.EW)
        
        self.stop_btn = ttk.Button(self.control_panel, text="Detener", command=self.stop_detection, state='disabled')
        self.stop_btn.grid(row=10, column=0, pady=(0, 20), sticky=tk.EW)
        
        # Configuración de columnas
        self.control_panel.columnconfigure(0, weight=1)
        
        # Manejar cambios en la selección de fuente
        self.source_var.trace_add('write', self.handle_source_change)
        
    def create_event_log(self):
        self.event_log = scrolledtext.ScrolledText(self.status_panel, width=40, height=15, state='disabled')
        self.event_log.pack(fill=tk.BOTH, expand=True)
        
        # Contador de detecciones
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
        if self.source_var.get() == "Archivo":
            self.file_entry.config(state='normal')
            self.browse_btn.config(state='normal')
        else:
            self.file_entry.config(state='disabled')
            self.browse_btn.config(state='disabled')
            self.file_entry.delete(0, tk.END)
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4;*.avi;*.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
    
    def load_model(self):
        try:
            proto = "models/MobileNetSSD_deploy.prototxt.txt"
            model = "models/MobileNetSSD_deploy.caffemodel"
            
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
        else:
            self.event_log.tag_add("info", "end-1c linestart", "end-1c lineend")
            self.event_log.tag_config("info", foreground="green")
        
        self.event_log.see(tk.END)
        self.event_log.config(state='disabled')
    
    def start_detection(self):
        if self.detecting:
            return
            
        # Configurar fuente de video
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
        
        # Reiniciar contadores
        for animal in self.REQ_CLASSES:
            self.detection_counters[animal].set("0")
        
        # Actualizar interfaz
        self.detecting = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_var.set("Detectando animales...")
        
        # Iniciar detección en un hilo separado
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
        alert_cooldown = 5  # Segundos entre alertas
        
        try:
            while self.detecting and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    self.log_event("Fin del video alcanzado", "warning")
                    break
                
                # Procesar frame
                frame = imutils.resize(frame, width=800)
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                
                self.net.setInput(blob)
                detections = self.net.forward()
                
                # Dibujar detecciones
                detections_in_frame = []
                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > float(self.threshold_var.get()):
                        idx = int(detections[0, 0, i, 1])
                        if self.CLASSES[idx] in self.REQ_CLASSES:
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            
                            # Dibujar rectángulo y etiqueta
                            color = (36, 255, 12)  # Verde
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                            
                            label = f"{self.CLASSES[idx]}: {confidence * 100:.1f}%"
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(frame, label, (startX, y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            detections_in_frame.append(self.CLASSES[idx])
                
                # Actualizar contadores
                for animal in set(detections_in_frame):
                    current = int(self.detection_counters[animal].get())
                    self.detection_counters[animal].set(str(current + 1))
                
                # Verificar intrusión (más de 15 detecciones en los últimos 36 frames)
                detection_history.append(1 if detections_in_frame else 0)
                if len(detection_history) > 36:
                    detection_history.pop(0)
                    
                    current_time = time.time()
                    if sum(detection_history) > 15 and not self.alarm_active and (current_time - last_alert_time) > alert_cooldown:
                        self.trigger_alarm()
                        last_alert_time = current_time
                
                # Mostrar frame en la interfaz
                self.update_video_display(frame)
                
                # Actualizar FPS
                self.fps.update()
                frame_count += 1
                if frame_count % 5 == 0:  # Actualizar FPS cada 5 frames
                    self.fps_var.set(f"FPS: {self.fps.fps():.2f}")
                
                # Pequeña pausa para no saturar la CPU
                time.sleep(0.01)
            
        except Exception as e:
            self.log_event(f"Error durante la detección: {str(e)}", "error")
            messagebox.showerror("Error", f"Ocurrió un error durante la detección:\n{str(e)}")
        
        finally:
            # Limpieza final
            if hasattr(self, 'fps'):
                self.fps.stop()
                
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            
            cv2.destroyAllWindows()
            
            self.detecting = False
            self.alarm_active = False
            
            # Actualizar interfaz en el hilo principal
            self.root.after(0, self.update_ui_after_stop)
        
    def update_ui_after_stop(self):
        """Actualiza la interfaz después de detener la detección"""
        self.stop_btn.config(state='disabled')
        self.start_btn.config(state='normal')
        self.status_var.set("Listo")
        self.log_event(f"Procesamiento completado. FPS promedio: {self.fps.fps():.2f}")
    
    def update_video_display(self, frame):
        # Convertir frame de OpenCV a formato compatible con Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Actualizar la imagen en el label
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def trigger_alarm(self):
        self.alarm_active = True
        self.log_event("¡Intrusión de animal detectada!", "warning")
        
        # Reproducir sonido de alarma (en un hilo separado)
        def play_alarm():
            try:
                playsound.playsound("alert/ringtone.mp3", block=False)
            except Exception as e:
                self.log_event(f"No se pudo reproducir la alarma: {str(e)}", "error")
            finally:
                self.alarm_active = False
        
        threading.Thread(target=play_alarm, daemon=True).start()
    
    def on_closing(self):
        if self.detecting:
            self.stop_detection()
            time.sleep(0.5)  # Pequeña espera para que el hilo se detenga
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    
    # Estilo moderno
    style = ttk.Style()
    style.theme_use('clam')  # Puedes probar con 'alt', 'default', 'clam' o 'vista'
    
    app = AnimalDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()