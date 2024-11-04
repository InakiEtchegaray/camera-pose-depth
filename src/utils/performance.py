import time
import logging
import psutil
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    fps: float
    cpu_percent: float
    memory_percent: float
    gpu_memory_used: Optional[float] = None
    processing_time: float = 0.0
    frame_count: int = 0
    dropped_frames: int = 0

class PerformanceMonitor:
    def __init__(self, history_size: int = 30):
        """
        Inicializa el monitor de rendimiento.
        
        Args:
            history_size: Número de muestras para calcular promedios
        """
        self.start_time = time.time()
        self.frame_times = deque(maxlen=history_size)
        self.processing_times = deque(maxlen=history_size)
        self.metrics_history = deque(maxlen=history_size)
        self.frame_count = 0
        self.dropped_frames = 0
        
        # Inicializar proceso para métricas del sistema
        self.process = psutil.Process()
        
    def start_frame(self) -> float:
        """Marca el inicio del procesamiento de un frame."""
        return time.time()
    
    def end_frame(self, start_time: float, success: bool = True) -> None:
        """
        Registra el fin del procesamiento de un frame.
        
        Args:
            start_time: Tiempo de inicio del procesamiento
            success: Si el frame se procesó exitosamente
        """
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        if success:
            self.frame_count += 1
        else:
            self.dropped_frames += 1
            
        self.frame_times.append(time.time())
    
    def get_metrics(self) -> PerformanceMetrics:
        """Obtiene las métricas actuales de rendimiento."""
        # Calcular FPS
        if len(self.frame_times) >= 2:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            fps = (len(self.frame_times) - 1) / time_diff if time_diff > 0 else 0
        else:
            fps = 0
            
        # Métricas del sistema
        try:
            cpu_percent = self.process.cpu_percent()
            memory_percent = self.process.memory_percent()
        except Exception as e:
            logger.error(f"Error al obtener métricas del sistema: {e}")
            cpu_percent = 0
            memory_percent = 0
            
        # Tiempo promedio de procesamiento
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0
        )
        
        metrics = PerformanceMetrics(
            fps=fps,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            processing_time=avg_processing_time,
            frame_count=self.frame_count,
            dropped_frames=self.dropped_frames
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def add_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Agrega un overlay con métricas al frame.
        
        Args:
            frame: Frame al que agregar el overlay
            
        Returns:
            Frame con overlay
        """
        metrics = self.get_metrics()
        
        # Crear overlay
        overlay = frame.copy()
        alpha = 0.4
        
        # Región para métricas
        cv2.rectangle(overlay, (10, 10), (250, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Texto de métricas
        metrics_text = [
            f"FPS: {metrics.fps:.1f}",
            f"CPU: {metrics.cpu_percent:.1f}%",
            f"MEM: {metrics.memory_percent:.1f}%",
            f"Process Time: {metrics.processing_time*1000:.1f}ms",
            f"Dropped Frames: {metrics.dropped_frames}"
        ]
        
        y = 30
        for text in metrics_text:
            cv2.putText(frame, text, (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 20
            
        return frame
    
    def log_metrics(self) -> None:
        """Registra las métricas actuales en el log."""
        metrics = self.get_metrics()
        logger.info(
            f"Performance - FPS: {metrics.fps:.1f}, "
            f"CPU: {metrics.cpu_percent:.1f}%, "
            f"MEM: {metrics.memory_percent:.1f}%, "
            f"Process Time: {metrics.processing_time*1000:.1f}ms, "
            f"Dropped: {metrics.dropped_frames}"
        )