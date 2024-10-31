import cv2
import logging
import numpy as np
from av import VideoFrame
from aiortc import MediaStreamTrack
from fractions import Fraction
import time
import torch
import gc

from depth_processor import DepthProcessor
from pose_processor import PoseProcessor
from utils.performance import PerformanceMonitor

logger = logging.getLogger(__name__)

class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        """Inicializa el procesador de video con todos sus componentes."""
        super().__init__()
        logger.info("Iniciando procesador de video")
        
        # Inicializar componentes
        self._init_cuda()
        self._init_camera()
        self._init_processors()
        self.pts = 0
        self.time_base = Fraction(1, 30)

    def _init_cuda(self):
        """Inicializa y configura CUDA si está disponible."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            logger.info(f"CUDA inicializado - GPU: {torch.cuda.get_device_name(0)}")

    def _init_camera(self):
        """Inicializa y configura la cámara web."""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")
        
        logger.info("Cámara inicializada correctamente")

    def _init_processors(self):
        """Inicializa los procesadores de pose, profundidad y rendimiento."""
        try:
            self.pose_processor = PoseProcessor()
            self.depth_processor = DepthProcessor()
            self.performance_monitor = PerformanceMonitor()
            logger.info("Procesadores inicializados correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar procesadores: {str(e)}")
            raise

    async def recv(self):
        """
        Recibe y procesa un frame de la cámara.
        
        Returns:
            VideoFrame: Frame procesado listo para WebRTC
        """
        try:
            start_time = time.time()
            
            # Capturar frame
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Error al leer frame")
                return None
            
            # Procesar frame
            frame = self.pose_processor.process_frame(frame)
            frame = await self.depth_processor.process_frame(frame)
            
            # Actualizar y mostrar métricas
            fps = self.performance_monitor.update(start_time)
            if fps is not None:
                logger.info(f"FPS: {fps:.1f}")
            
            # Añadir overlay de rendimiento
            frame = self.performance_monitor.add_overlay(frame)
            
            # Convertir para WebRTC
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            video_frame.pts = self.pts
            video_frame.time_base = self.time_base
            
            self.pts += 1
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
            
            gc.collect()
            logger.info("Recursos liberados correctamente")
        except Exception as e:
            logger.error(f"Error al liberar recursos: {str(e)}")