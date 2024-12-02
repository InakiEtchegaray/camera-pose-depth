import time
import logging
import asyncio
from fractions import Fraction
from aiortc import MediaStreamTrack
from av import VideoFrame
import supervision as sv
from ultralytics import YOLO
import numpy as np
import cv2
import torch
from config import config

logger = logging.getLogger(__name__)

class SupervisionTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, config_params=None):
        super().__init__()
        self.config_params = config_params or {}
        self.frame_count = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.time_base = Fraction(1, 30)

        # Verificar GPU
        logger.info(f"Usando GPU: {torch.cuda.is_available()}, Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        
        self.setup_detector()
        self.setup_camera()
        logger.info("SupervisionTransformTrack iniciado")

    def setup_detector(self):
        try:
            self.model = YOLO(config.SUPERVISION.MODEL_PATH)
            self.box_annotator = sv.BoxAnnotator(thickness=2)
        except Exception as e:
            logger.error(f"Error configurando detector: {e}")
            raise

    def setup_camera(self):
        """Inicializa la cámara"""
        try:
            logger.info("Iniciando cámara...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                raise RuntimeError("No se pudo acceder a la cámara")

            width = self.config_params.get('width', 640)
            height = self.config_params.get('height', 480)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Leer frame de prueba
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("No se pudo leer frame de prueba")
            
            logger.info("Cámara iniciada correctamente")
                
        except Exception as e:
            logger.error(f"Error iniciando cámara: {e}")
            if hasattr(self, 'cap'):
                self.cap.release()
            raise

    async def recv(self):
        try:
            if not self.cap.isOpened():
                self.setup_camera()
                
            ret, img = self.cap.read()
            if not ret:
                time.sleep(0.1)
                ret, img = self.cap.read()
                if not ret:
                    raise RuntimeError("Error al leer frame de la cámara")

            # Control de frecuencia de logs
            current_time = time.time()
            should_log = current_time - self.last_log_time >= 1.0

            if should_log:
                self.last_log_time = current_time
                results = self.model(
                    img,
                    conf=config.SUPERVISION.CONFIDENCE_THRESHOLD,
                    device=config.SUPERVISION.DEVICE
                )[0]
            else:
                results = self.model(
                    img,
                    conf=config.SUPERVISION.CONFIDENCE_THRESHOLD,
                    device=config.SUPERVISION.DEVICE,
                    verbose=False
                )[0]

            # Convertir resultados a detecciones
            boxes = results.boxes.xyxy
            if len(boxes) > 0:
                # Asegurar que los tensores estén en CPU antes de convertir a numpy
                boxes = boxes.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                
                detections = sv.Detections(
                    xyxy=boxes,
                    confidence=confidences,
                    class_id=class_ids
                )
                
                # Filtrar solo personas (clase 0)
                mask = [cls == 0 for cls in detections.class_id]
                detections = detections[mask]
                
                if len(detections) > 0:
                    # Dibujar detecciones
                    for i, (box, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
                        x1, y1, x2, y2 = box.astype(int)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        label = f"Persona {i+1} ({conf:.2f})"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        
                        cv2.rectangle(
                            img,
                            (x1, y1 - 25),
                            (x1 + text_size[0], y1),
                            (0, 0, 0),
                            -1
                        )
                        
                        cv2.putText(
                            img,
                            label,
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2
                        )

            await asyncio.sleep(0.001)

            # Crear nuevo frame
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = self.frame_count
            new_frame.time_base = self.time_base
            
            self.frame_count += 1
            return new_frame
            
        except Exception as e:
            logger.error(f"Error en recv: {e}")
            if not self.cap.isOpened():
                self.setup_camera()
            raise

    def get_performance_metrics(self):
        try:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

            return {
                'fps': round(current_fps, 1),
                'status': 'connected'
            }
        except Exception as e:
            logger.error(f"Error en métricas: {e}")
            return {
                'fps': 0,
                'status': 'error'
            }

    def stop(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            logger.info("Cámara liberada")
        if hasattr(self, 'model'):
            del self.model
        super().stop()