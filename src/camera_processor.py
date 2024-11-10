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
        self.frame_intervals = deque(maxlen=window_size)
        self.last_time = time.time()
        self._fps = 0

    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Calcular intervalo entre frames
        if len(self.frame_times) > 1:
            interval = self.frame_times[-1] - self.frame_times[-2]
            self.frame_intervals.append(interval)
        
        # Calcular FPS usando una ventana móvil
        if len(self.frame_intervals) > 0:
            avg_interval = sum(self.frame_intervals) / len(self.frame_intervals)
            self._fps = 1.0 / avg_interval if avg_interval > 0 else 0
        
        return self._fps

    @property
    def fps(self):
        return self._fps

class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, config):
        """Inicializa el procesador de video con una configuración específica."""
        super().__init__()
        self.config = config
        logger.info("Iniciando procesador de video")
        
        # Inicializar variables
        self._camera_lock = asyncio.Lock()
        self._last_frame = None
        self.cap = None
        self.pts = 0
        self.time_base = Fraction(1, 30)
        self.frame_count = 0
        self.fps_counter = FPSCounter()
        self.last_metrics_update = 0
        self.metrics_update_interval = 0.5  # Actualizar métricas cada 0.5 segundos
        self._supported_resolutions = []
        self.on_metrics_update = None
        
        # Inicializar cámara y procesadores
        if not self._init_camera_with_retry():
            raise RuntimeError("No se pudo inicializar la cámara después de múltiples intentos")
        
        self._init_processors()

    def _get_supported_resolutions(self):
        """Detecta las resoluciones soportadas por la cámara."""
        if self._supported_resolutions:
            return self._supported_resolutions

        supported = []
        test_resolutions = [
            (1920, 1080),  # Full HD
            (1280, 720),   # HD
            (1024, 768),   # XGA
            (800, 600),    # SVGA
            (640, 480),    # VGA
        ]
        
        original_resolution = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        
        logger.info("Detectando resoluciones soportadas...")
        
        # Configurar MJPG codec para mejor soporte de resoluciones
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        for width, height in test_resolutions:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verificar resolución real
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Leer frame para verificar
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                
                if frame_width > 0 and frame_height > 0:
                    resolution = (frame_width, frame_height)
                    if resolution not in supported:
                        supported.append(resolution)
                        logger.info(f"Resolución soportada: {frame_width}x{frame_height}")

        # Restaurar resolución original
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_resolution[1])
        
        # Eliminar duplicados y ordenar por tamaño
        self._supported_resolutions = sorted(set(supported), 
                                          key=lambda x: x[0] * x[1], 
                                          reverse=True)
        
        return self._supported_resolutions

    def _find_closest_resolution(self, target_width, target_height):
        """Encuentra la resolución soportada más cercana a la objetivo."""
        if not self._supported_resolutions:
            self._get_supported_resolutions()
        
        target_ratio = target_width / target_height
        target_pixels = target_width * target_height
        
        def resolution_score(res):
            ratio = res[0] / res[1]
            pixels = res[0] * res[1]
            ratio_diff = abs(ratio - target_ratio)
            pixel_diff = abs(pixels - target_pixels)
            return ratio_diff + pixel_diff / (1920 * 1080)  # Normalizar diferencia de píxeles
        
        return min(self._supported_resolutions, key=resolution_score)

    def _find_available_camera(self):
        """Busca una cámara disponible."""
        available_cameras = []
        for i in range(10):  # Probar los primeros 10 índices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        if available_cameras:
            logger.info(f"Cámaras disponibles: {available_cameras}")
            return available_cameras[0]
        return None

    def _init_camera_with_retry(self, max_attempts=3, delay=1.0):
        """Intenta inicializar la cámara con múltiples intentos."""
        for attempt in range(max_attempts):
            try:
                logger.info(f"Intento {attempt + 1} de inicializar la cámara")
                
                camera_id = self._find_available_camera()
                if camera_id is None:
                    logger.error("No se encontraron cámaras disponibles")
                    time.sleep(delay)
                    continue

                self.cap = cv2.VideoCapture(camera_id)
                if not self.cap.isOpened():
                    logger.error(f"No se pudo abrir la cámara {camera_id}")
                    time.sleep(delay)
                    continue

                if not self._set_camera_properties():
                    logger.error("No se pudieron configurar las propiedades de la cámara")
                    self.cap.release()
                    time.sleep(delay)
                    continue

                # Verificar que podemos leer frames
                for _ in range(5):  # Intentar leer varios frames
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self._last_frame = frame.copy()
                        logger.info("Cámara inicializada exitosamente")
                        return True
                    time.sleep(0.1)

                logger.error("No se pudo leer frame inicial")
                self.cap.release()
                time.sleep(delay)

            except Exception as e:
                logger.error(f"Error al inicializar cámara en intento {attempt + 1}: {str(e)}")
                if self.cap:
                    self.cap.release()
                time.sleep(delay)

        return False

    def _set_camera_properties(self):
        """Configura las propiedades de la cámara."""
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            target_width = self.config.get('width', 640)
            target_height = self.config.get('height', 480)
            
            # Encontrar la resolución más cercana disponible
            best_width, best_height = self._find_closest_resolution(target_width, target_height)
            
            logger.info(f"Resolución objetivo: {target_width}x{target_height}")
            logger.info(f"Mejor resolución disponible: {best_width}x{best_height}")
            
            # Establecer la resolución
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_height)
            
            # Otras configuraciones
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Actualizar configuración
            self.config['width'] = best_width
            self.config['height'] = best_height
            
            # Verificar configuración
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False
            
            actual_width = frame.shape[1]
            actual_height = frame.shape[0]
            logger.info(f"Resolución final establecida: {actual_width}x{actual_height}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error al configurar propiedades de la cámara: {str(e)}")
            return False

    def _init_processors(self):
        """Inicializa los procesadores de pose y profundidad."""
        try:
            self.pose_processor = PoseProcessor()
            self.depth_processor = DepthProcessor()
            logger.info("Procesadores inicializados correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar procesadores: {str(e)}")
            raise

    async def update_config(self, new_config):
        """Actualiza la configuración en tiempo real."""
        try:
            logger.info(f"Actualizando configuración: {new_config}")
            
            async with self._camera_lock:
                old_config = self.config.copy()
                self.config.update(new_config)
                
                if ('width' in new_config or 'height' in new_config):
                    if not self._init_camera_with_retry():
                        logger.error("Error al cambiar resolución, volviendo a configuración anterior")
                        self.config = old_config
                        self._init_camera_with_retry()
            
            logger.info("Configuración actualizada exitosamente")
            
        except Exception as e:
            logger.error(f"Error al actualizar configuración: {e}")

    async def recv(self):
        """Recibe y procesa un frame de la cámara."""
        try:
            async with self._camera_lock:
                if self.cap is None or not self.cap.isOpened():
                    if not self._init_camera_with_retry():
                        raise RuntimeError("Cámara no disponible")

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    if self._last_frame is not None:
                        frame = self._last_frame.copy()
                    else:
                        raise RuntimeError("No hay frame disponible")
                else:
                    self._last_frame = frame.copy()

            # Procesar frame
            if self.config.get('pose_enabled', True):
                frame = self.pose_processor.process_frame(frame)
            
            if self.config.get('depth_enabled', True):
                frame = await self.depth_processor.process_frame(frame)

            # Convertir para WebRTC
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            new_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            
            # Actualizar timestamps
            pts, time_base = await self.next_timestamp()
            new_frame.pts = pts
            new_frame.time_base = time_base

            # Actualizar FPS y métricas
            current_fps = self.fps_counter.update()
            self.frame_count += 1
            
            # Actualizar métricas periódicamente
            current_time = time.time()
            if current_time - self.last_metrics_update >= self.metrics_update_interval:
                if self.on_metrics_update:
                    try:
                        await self.on_metrics_update({
                            'fps': current_fps,
                            'frame_count': self.frame_count,
                            'resolution': f"{frame.shape[1]}x{frame.shape[0]}",
                            'processing': {
                                'pose': self.config.get('pose_enabled', True),
                                'depth': self.config.get('depth_enabled', True)
                            }
                        })
                    except Exception as e:
                        logger.error(f"Error al actualizar métricas: {e}")
                self.last_metrics_update = current_time

            return new_frame
            
        except Exception as e:
            logger.error(f"Error en recv: {str(e)}")
            return None

    async def next_timestamp(self):
        """Genera el siguiente timestamp para el frame."""
        pts = self.pts
        self.pts += 1
        return pts, self.time_base

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