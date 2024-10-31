import time
import logging
import torch
import cv2

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self, max_samples=30, fps_update_interval=1.0):
        """
        Inicializa el monitor de rendimiento.
        
        Args:
            max_samples (int): Número máximo de muestras para promediar
            fps_update_interval (float): Intervalo de actualización de FPS en segundos
        """
        self.max_samples = max_samples
        self.fps_update_interval = fps_update_interval
        self._init_metrics()

    def _init_metrics(self):
        """Inicializa las métricas de rendimiento."""
        self.processing_times = []
        self.frame_count = 0
        self.total_frames = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.last_fps_update = time.time()

    def update(self, start_time):
        """
        Actualiza las métricas con un nuevo frame.
        
        Args:
            start_time (float): Tiempo de inicio del procesamiento del frame
            
        Returns:
            float: FPS actuales si se actualizaron, None en caso contrario
        """
        current_time = time.time()
        process_time = current_time - start_time
        
        # Actualizar contadores
        self.frame_count += 1
        self.total_frames += 1
        self.processing_times.append(process_time)
        
        # Mantener solo las últimas N muestras
        if len(self.processing_times) > self.max_samples:
            self.processing_times.pop(0)
        
        # Actualizar FPS si corresponde
        if current_time - self.last_fps_update >= self.fps_update_interval:
            self.current_fps = self.frame_count / (current_time - self.fps_start_time)
            self.frame_count = 0
            self.fps_start_time = current_time
            self.last_fps_update = current_time
            return self.current_fps
            
        return None

    def get_metrics(self):
        """
        Obtiene las métricas actuales.
        
        Returns:
            dict: Diccionario con las métricas actuales
        """
        avg_process_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        metrics = {
            'fps': self.current_fps,
            'avg_process_time': avg_process_time * 1000,  # convertir a ms
            'total_frames': self.total_frames
        }
        
        # Añadir métricas de GPU si está disponible
        if torch.cuda.is_available():
            metrics.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**2     # MB
            })
            
        return metrics

    def add_overlay(self, frame):
        """
        Añade una superposición con métricas al frame.
        
        Args:
            frame: Frame al que añadir las métricas
            
        Returns:
            frame: Frame con las métricas superpuestas
        """
        metrics = self.get_metrics()
        
        # Configuración del texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        color = (255, 255, 255)
        
        # Posición inicial
        y_position = 30
        x_position = 10
        
        # Función helper para añadir texto
        def add_text(text):
            nonlocal y_position
            cv2.putText(frame, text, (x_position, y_position), font, scale, color, thickness)
            y_position += 25

        # Añadir métricas
        add_text(f"FPS: {metrics['fps']:.1f}")
        add_text(f"Proc time: {metrics['avg_process_time']:.1f}ms")
        
        if 'gpu_name' in metrics:
            add_text(f"GPU: {metrics['gpu_name']}")
            add_text(f"GPU Mem: {metrics['gpu_memory_allocated']:.1f}MB")
        
        return frame