import cv2
import imutils
import numpy as np
import time
from collections import Counter
from threading import Thread, Event
from imutils.video import FPS
import playsound
import os
import tkinter as tk
from tkinter import ttk, BooleanVar, Checkbutton, Button, filedialog, messagebox, Frame, Label, scrolledtext
from twilio.rest import Client

# Configuration for Twilio
FARM_OWNER_NUMBER = "+525663724981"
TWILIO_ACCOUNT_SID = "ACde00b9b8793ce70ac800d1add00f041d"
TWILIO_AUTH_TOKEN = "4dccb685a8ac0bab74ec29d15f169850"
TWILIO_PHONE_NUMBER = "+18582640158"

# Paths
PROTO_PATH = "C:/Users/Angel/Desktop/files/models/MobileNetSSD_deploy.prototxt.txt"
MODEL_PATH = "C:/Users/Angel/Desktop/files/models/MobileNetSSD_deploy.caffemodel"
SIREN_PATH = "siren/Siren.wav"

# Classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
           "chair", "cow", "dining-table", "dog", "horse", "motorbike", "person", 
           "potted plant", "sheep", "sofa", "train", "monitor"]
REQ_CLASSES = ["bird", "cat", "cow", "dog", "horse", "sheep"]

# Global variables
detection_report = []
animal_timers = {}
animal_vars = {}
stop_event = Event()
detection_active = False
video_capture = None
alarm_active = False
event_log = None

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

def load_model(proto_path, model_path):
    if not os.path.exists(proto_path):
        raise FileNotFoundError(f"Prototxt file not found: {proto_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Caffe model file not found: {model_path}")
    return cv2.dnn.readNetFromCaffe(proto_path, model_path)

def send_sms(phone_number, detected_time):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"Alert! Animal intrusion detected on your farm at {detected_time}.",
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        print("[INFO] SMS sent successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send SMS: {e}")
        return False

def play_siren(siren_path):
    if not os.path.exists(siren_path):
        print(f"[ERROR] Siren file not found: {siren_path}")
        return
    try:
        playsound.playsound(siren_path, block=False)
    except Exception as e:
        print(f"[ERROR] Failed to play siren: {e}")

def log_event(message, level="info"):
    if event_log:
        try:
            event_log.config(state='normal')
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            event_log.insert(tk.END, f"[{timestamp}] {message}\n")
            
            if level == "error":
                event_log.tag_add("error", "end-1c linestart", "end-1c lineend")
                event_log.tag_config("error", foreground="red")
            elif level == "warning":
                event_log.tag_add("warning", "end-1c linestart", "end-1c lineend")
                event_log.tag_config("warning", foreground="orange")
            
            event_log.see(tk.END)
            event_log.config(state='disabled')
        except tk.TclError:
            # Si la ventana ya está cerrada, solo imprime en consola
            print(f"[{level.upper()}] {message}")
    else:
        print(f"[{level.upper()}] {message}")

class AnimalDetectionApp:
    def __init__(self):
        self.root = None
        self.start_btn = None
        self.stop_btn = None
        self.setup_filter_ui()
    
    def setup_filter_ui(self):
        global animal_vars, event_log
        
        self.root = tk.Tk()
        self.root.title("Sistema de Detección de Animales")
        self.root.geometry("600x700")
        
        # Inicializar las variables Tkinter
        animal_vars = {animal: BooleanVar(value=True) for animal in REQ_CLASSES}
        
        # Configurar estilos personalizados para los botones
        style = ttk.Style()
        
        # Estilo para botón de inicio 
        style.configure("Start.TButton", 
                      fontcolor = "Black",
                      background="#4CAF50",
                      font=("Arial", 12, "bold"),
                      padding=10)
        
        # Estilo para botón de detener 
        style.configure("Stop.TButton", 
                      fontcolor = "Black",
                      background="#F44336",
                      font=("Arial", 12, "bold"),
                      padding=10)
        
        # Estilo para botón de exportar 
        style.configure("Export.TButton", 
                      fontcolor = "Black",
                      background="#2196F3",
                      font=("Arial", 12, "bold"),
                      padding=10)
        
        # Estilo para frames
        style.configure("TFrame", background="#f0f0f0")
        
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        # Panel de configuración
        config_frame = ttk.LabelFrame(main_frame, text="Configuración", padding=10)
        config_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(config_frame, text="Selecciona los animales a detectar:", 
                 font=("Arial", 12)).pack(anchor="w", pady=(0, 10))
        
        # Frame con scroll para checkboxes
        canvas = tk.Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        for animal in REQ_CLASSES:
            cb = Checkbutton(scroll_frame, 
                            text=animal.capitalize(), 
                            variable=animal_vars[animal],
                            font=("Arial", 10),
                            bg="#f0f0f0")
            cb.pack(anchor="w", padx=10, pady=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Panel de registro de eventos
        log_frame = ttk.LabelFrame(main_frame, text="Registro de Eventos", padding=10)
        log_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        event_log = scrolledtext.ScrolledText(log_frame, width=60, height=10, state='disabled')
        event_log.pack(fill="both", expand=True)
        
        # Botones de acción
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x")
        
        self.start_btn = ttk.Button(
            btn_frame,
            text="▶ INICIAR DETECCIÓN",
            command=self.start_detection,
            style="Start.TButton"
        )
        self.start_btn.pack(side="left", expand=True, padx=5)
        
        export_btn = ttk.Button(
            btn_frame,
            text="📄 EXPORTAR INFORME",
            command=self.export_report,
            style="Export.TButton"
        )
        export_btn.pack(side="left", expand=True, padx=5)
        
        self.stop_btn = ttk.Button(
            btn_frame,
            text="⏹ DETENER",
            command=self.stop_detection,
            style="Stop.TButton",
            state='disabled'
        )
        self.stop_btn.pack(side="left", expand=True, padx=5)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def start_detection(self):
        global detection_active
        detection_active = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        try:
            net = load_model(PROTO_PATH, MODEL_PATH)
            log_event("Modelo cargado correctamente")
            
            # Iniciar detección en un hilo separado
            detection_thread = Thread(target=self.run_detection, args=(net,), daemon=True)
            detection_thread.start()
            
        except Exception as e:
            log_event(f"No se pudo iniciar la detección: {str(e)}", "error")
            messagebox.showerror("Error", f"No se pudo iniciar la detección:\n{str(e)}")
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            cleanup_resources()
    
    def stop_detection(self):
        global detection_active
        detection_active = False
        log_event("Detección detenida por el usuario")
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
    
    def run_detection(self, net):
        global video_capture, alarm_active
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            log_event("No se pudo acceder a la cámara", "error")
            self.root.after(0, lambda: [
                self.start_btn.config(state='normal'),
                self.stop_btn.config(state='disabled')
            ])
            return
        
        fps = FPSCounter().start()
        detection_history = []
        last_alert_time = 0
        alert_cooldown = 5  # segundos
        
        try:
            while detection_active:
                ret, frame = video_capture.read()
                if not ret:
                    log_event("No se pudo capturar el frame de la cámara", "error")
                    break
                
                frame = imutils.resize(frame, width=800)
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                
                net.setInput(blob)
                detections = net.forward()
                
                detections_in_frame = self.process_frame(frame, detections, h, w)
                
                cv2.imshow("Detección de Animales", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.root.after(0, self.stop_detection)
                    break
                
                # Lógica de alarma
                detection_history.append(1 if detections_in_frame else 0)
                if len(detection_history) > 36:
                    detection_history.pop(0)
                    
                    current_time = time.time()
                    if sum(detection_history) > 15 and not alarm_active and (current_time - last_alert_time) > alert_cooldown:
                        detected_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        log_event(f"¡Intrusión de animal detectada a las {detected_time}!", "warning")
                        play_siren(SIREN_PATH)
                        send_sms(FARM_OWNER_NUMBER, detected_time)
                        alarm_active = True
                        last_alert_time = current_time
                
                fps.update()
            
        except Exception as e:
            log_event(f"Error durante la detección: {str(e)}", "error")
            print(f"[ERROR] {str(e)}")
        finally:
            fps.stop()
            log_event(f"Procesamiento completado. FPS promedio: {fps.fps():.2f}")
            print(f"[INFO] FPS promedio: {fps.fps():.2f}")
            cleanup_resources()
            self.root.after(0, lambda: [
                self.start_btn.config(state='normal'),
                self.stop_btn.config(state='disabled')
            ])
    
    def process_frame(self, frame, detections, h, w):
        detections_in_frame = []
        
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                animal = CLASSES[idx]
                
                if animal in REQ_CLASSES and animal_vars[animal].get():
                    detections_in_frame.append(animal)
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = f"{animal}: {confidence * 100:.1f}%"
                    cv2.putText(frame, label, (startX, startY - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    if animal not in animal_timers:
                        animal_timers[animal] = {"start": time.time(), "end": None}
                        log_event(f"Detección de {animal} a las {time.strftime('%H:%M:%S')}")
        
        self.update_detection_times(detections_in_frame)
        return detections_in_frame
    
    def update_detection_times(self, current_detections):
        for animal in list(animal_timers.keys()):
            if animal not in current_detections and animal_timers[animal]["end"] is None:
                animal_timers[animal]["end"] = time.time()
                detection_report.append({
                    "animal": animal,
                    "hora": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duracion": animal_timers[animal]["end"] - animal_timers[animal]["start"]
                })
                del animal_timers[animal]
    
    def export_report(self):
        if not detection_report:
            log_event("No se detectaron animales para generar informe", "warning")
            messagebox.showwarning("Advertencia", "No hay datos para exportar.")
            return
        
        # Filtrar detecciones según los checkboxes activos
        filtered_detections = [
            entry for entry in detection_report
            if animal_vars[entry["animal"]].get()
        ]
        
        if not filtered_detections:
            log_event("No hay datos con los filtros actuales para generar informe", "warning")
            messagebox.showwarning("Advertencia", "No hay datos con los filtros actuales.")
            return
        
        report_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Archivo de texto", "*.txt"), ("Todos los archivos", "*.*")],
            title="Guardar informe de detecciones"
        )
        
        if report_path:
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write("=== INFORME DE DETECCIONES ===\n\n")
                    f.write(f"Animales incluidos: {', '.join([a for a in REQ_CLASSES if animal_vars[a].get()])}\n\n")
                    
                    for entry in filtered_detections:
                        f.write(f"Animal: {entry['animal']}\n")
                        f.write(f"Hora: {entry['hora']}\n")
                        f.write(f"Duración: {entry['duracion']:.2f} segundos\n")
                        f.write("-" * 40 + "\n")
                
                log_event(f"Informe exportado correctamente: {report_path}")
                messagebox.showinfo("Éxito", f"Informe guardado en:\n{report_path}")
                os.startfile(report_path)
            except Exception as e:
                log_event(f"Error al exportar informe: {str(e)}", "error")
                messagebox.showerror("Error", f"No se pudo guardar el informe:\n{str(e)}")
    
    def on_closing(self):
        global detection_active
        detection_active = False
        
        if video_capture is not None and video_capture.isOpened():
            video_capture.release()
        
        cv2.destroyAllWindows()
        self.root.destroy()

def cleanup_resources():
    global detection_active, video_capture, alarm_active
    detection_active = False
    alarm_active = False
    
    if video_capture is not None:
        video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        app = AnimalDetectionApp()
    except Exception as e:
        print(f"[CRITICAL ERROR] {str(e)}")
    finally:
        cleanup_resources()
