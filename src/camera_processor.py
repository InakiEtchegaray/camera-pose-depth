import cv2
import logging
import numpy as np
from av import VideoFrame
from aiortc import MediaStreamTrack
from fractions import Fraction
import time
import torch
import asyncio
from collections import deque

from depth_processor import DepthProcessor
from pose_processor import PoseProcessor

logger = logging.getLogger(__name__)

class FPSCounter:
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()
        self.fps = 0

    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        
        if len(self.frame_times) >= 2:
            self.fps = len(self.frame_times) / sum(self.frame_times)
        
        return self.fps

class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, config):
        super().__init__()
        self.config = config
        logger.info("Iniciando procesador de video")
        
        self._init_camera()
        self._init_processors()
        self.pts = 0
        self.time_base = Fraction(1, 30)
        self._frame_count = 0
        self.fps_counter = FPSCounter()
        self.on_metrics_update = None  # Callback para métricas

    def _init_camera(self):
        """Inicializa y configura la cámara web."""
        try:
            self.cap = cv2.VideoCapture(0)
            self._set_camera_properties()
            
            if not self.cap.isOpened():
                raise RuntimeError("No se pudo abrir la cámara")
            
            logger.info("Cámara inicializada correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar cámara: {str(e)}")
            raise

    def _set_camera_properties(self):
        """Configura las propiedades de la cámara."""
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['height'])
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Resolución establecida: {actual_width}x{actual_height}")
            
        except Exception as e:
            logger.error(f"Error al configurar propiedades de la cámara: {str(e)}")

    def _init_processors(self):
        """Inicializa los procesadores de pose y profundidad."""
        try:
            self.pose_processor = PoseProcessor()
            self.depth_processor = DepthProcessor()
            logger.info("Procesadores inicializados correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar procesadores: {str(e)}")
            raise

    def _update_metrics(self):
        """Actualiza y envía las métricas."""
        fps = self.fps_counter.update()
        if self.on_metrics_update:
            metrics = {
                'fps': fps,
                'frame_count': self._frame_count,
                'processing_active': {
                    'pose': self.config.get('pose_enabled', True),
                    'depth': self.config.get('depth_enabled', True)
                },
                'resolution': {
                    'width': self.config['width'],
                    'height': self.config['height']
                }
            }
            try:
                asyncio.create_task(self.on_metrics_update(metrics))
            except Exception as e:
                logger.error(f"Error al enviar métricas: {e}")

    async def recv(self):
        """Recibe y procesa un frame de la cámara."""
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Error al leer frame")
                return None

            self._frame_count += 1
            process_start_time = time.time()

            # Aplicar procesamiento según configuración
            if self.config.get('pose_enabled', True):
                frame = self.pose_processor.process_frame(frame)
            
            if self.config.get('depth_enabled', True):
                frame = await self.depth_processor.process_frame(frame)

            # Convertir para WebRTC
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            new_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            
            pts, time_base = await self.next_timestamp()
            new_frame.pts = pts
            new_frame.time_base = time_base

            # Actualizar métricas
            self._update_metrics()

            return new_frame
            
        except Exception as e:
            logger.error(f"Error en recv: {str(e)}")
            return None

    async def next_timestamp(self):
        """Genera el siguiente timestamp."""
        pts = self.pts
        self.pts += 1
        return pts, self.time_base

    async def update_config(self, new_config):
        """Actualiza la configuración en tiempo real."""
        try:
            logger.info(f"Actualizando configuración: {new_config}")
            
            # Si cambia la resolución, reiniciar la cámara
            if 'width' in new_config and 'height' in new_config:
                self.cap.release()
                await asyncio.sleep(0.1)
                self.cap = cv2.VideoCapture(0)
                self.config.update(new_config)
                self._set_camera_properties()
            else:
                self.config.update(new_config)
            
            logger.info("Configuración actualizada exitosamente")
            
        except Exception as e:
            logger.error(f"Error al actualizar configuración: {e}")

    def __del__(self):
        """Limpia los recursos."""
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Recursos liberados correctamente")
        except Exception as e:
            logger.error(f"Error al liberar recursos: {str(e)}")