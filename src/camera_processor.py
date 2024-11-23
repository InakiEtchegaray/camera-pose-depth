import cv2
import logging
import numpy as np
from av import VideoFrame
from aiortc import MediaStreamTrack
from fractions import Fraction
import time
import torch
from collections import deque
import asyncio
import psutil

from depth_processor import DepthProcessor
from pose_processor import PoseProcessor

logger = logging.getLogger(__name__)

class FPSCounter:
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()

    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time

        if len(self.frame_times) > 1:
            return len(self.frame_times) / sum(self.frame_times)
        return 0

class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, config):
        """Inicializa el procesador de video con una configuración específica."""
        super().__init__()
        self.config = config
        logger.info("Iniciando procesador de video")
        
        # Contadores y métricas
        self.fps_counter = FPSCounter()
        self.frame_count = 0
        self.processing_times = deque(maxlen=30)
        self._lock = asyncio.Lock()

        self._init_camera()
        self._init_processors()
        self.pts = 0
        self.time_base = Fraction(1, 30)

    def _init_camera(self):
        """Inicializa y configura la cámara web."""
        try:
            self.cap = cv2.VideoCapture(0)
            self._configure_camera()
            
            if not self.cap.isOpened():
                raise RuntimeError("No se pudo abrir la cámara")
            
            logger.info("Cámara inicializada correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar cámara: {str(e)}")
            raise

    def _configure_camera(self):
        """Configura las propiedades de la cámara."""
        width = self.config.get('width', 640)
        height = self.config.get('height', 480)
        
        # Intentar establecer la resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Verificar la resolución actual
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width != width or actual_height != height:
            logger.warning(f"Resolución solicitada no disponible: {width}x{height}")
            logger.warning(f"Usando resolución: {actual_width}x{actual_height}")
            self.config['width'] = actual_width
            self.config['height'] = actual_height

        # Configurar FPS y buffer
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    def _init_processors(self):
        """Inicializa los procesadores de pose y profundidad."""
        try:
            self.pose_processor = PoseProcessor()
            self.depth_processor = DepthProcessor()
            logger.info("Procesadores inicializados correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar procesadores: {str(e)}")
            raise

    def update_config(self, new_config):
        """Actualiza la configuración en tiempo real."""
        try:
            logger.info(f"Actualizando configuración: {new_config}")
            should_reinit = False

            if ('width' in new_config or 'height' in new_config) and \
               (new_config.get('width') != self.config.get('width') or \
                new_config.get('height') != self.config.get('height')):
                should_reinit = True

            self.config.update(new_config)

            if should_reinit:
                self.cap.release()
                time.sleep(0.1)
                self._init_camera()

            logger.info("Configuración actualizada correctamente")
            
        except Exception as e:
            logger.error(f"Error al actualizar configuración: {e}")
            raise

    def get_performance_metrics(self):
        """Obtiene métricas de rendimiento."""
        try:
            fps = self.fps_counter.update()
            avg_processing_time = (
                sum(self.processing_times) / len(self.processing_times)
                if self.processing_times else 0
            )

            metrics = {
                'fps': fps,
                'latency': avg_processing_time * 1000,  # convertir a ms
                'frame_count': self.frame_count
            }

            # Agregar métricas de GPU si está disponible
            if torch.cuda.is_available():
                metrics.update({
                    'gpu_usage': torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 
                    if torch.cuda.max_memory_allocated() > 0 else 0
                })

            # Métricas de CPU
            metrics['cpu_usage'] = psutil.Process().cpu_percent()

            return metrics
        except Exception as e:
            logger.error(f"Error al obtener métricas: {e}")
            return {}

    async def recv(self):
        """Recibe y procesa un frame de la cámara."""
        start_time = time.time()
        
        try:
            async with self._lock:
                if not self.cap.isOpened():
                    logger.error("Cámara no disponible")
                    return None

                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Error al leer frame")
                    return None

                self.frame_count += 1

                # Aplicar procesamiento según configuración
                if self.config.get('pose_enabled', True):
                    frame = self.pose_processor.process_frame(frame)
                
                if self.config.get('depth_enabled', True):
                    frame = await self.depth_processor.process_frame(frame)

                # Convertir para WebRTC
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                
                pts, time_base = self.pts, self.time_base
                video_frame.pts = pts
                video_frame.time_base = time_base
                self.pts += 1

                # Actualizar métricas
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)

                return video_frame
                
        except Exception as e:
            logger.error(f"Error en recv: {str(e)}")
            return None

    def __del__(self):
        """Limpia los recursos utilizados."""
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Recursos liberados correctamente")
        except Exception as e:
            logger.error(f"Error al liberar recursos: {str(e)}")