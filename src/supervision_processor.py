import time
import logging
import asyncio
from fractions import Fraction
from aiortc import MediaStreamTrack, RTCDataChannel
from av import VideoFrame
import supervision as sv
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import json
from dataclasses import dataclass
from typing import List, Optional
from config import config

logger = logging.getLogger(__name__)

@dataclass
class DetectionAlert:
    timestamp: float
    person_count: int
    area_occupied: bool
    confidence: float

class AlertManager:
    def __init__(self):
        self.data_channels: List[RTCDataChannel] = []
        self.last_alert: Optional[DetectionAlert] = None
        
    def add_data_channel(self, dc: RTCDataChannel):
        self.data_channels.append(dc)
        logger.info("Nuevo canal de datos añadido")
        
    def remove_data_channel(self, dc: RTCDataChannel):
        if dc in self.data_channels:
            self.data_channels.remove(dc)
            logger.info("Canal de datos removido")
            
    async def send_alert(self, alert: DetectionAlert):
        if not self.data_channels:
            return
            
        # Solo enviar alerta si es diferente a la última
        if (self.last_alert is None or
            alert.area_occupied != self.last_alert.area_occupied or
            alert.person_count != self.last_alert.person_count):
            
            alert_data = {
                "type": "detection_alert",
                "timestamp": alert.timestamp,
                "personCount": alert.person_count,
                "areaOccupied": alert.area_occupied,
                "confidence": alert.confidence
            }
            
            # Enviar a todos los canales de datos activos
            message = json.dumps(alert_data)
            for dc in self.data_channels.copy():
                try:
                    if dc.readyState == "open":
                        dc.send(message)
                    else:
                        self.remove_data_channel(dc)
                except Exception as e:
                    logger.error(f"Error enviando alerta por DataChannel: {e}")
                    self.remove_data_channel(dc)
            
            self.last_alert = alert

class SupervisionTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, config_params=None):
        super().__init__()
        self.config_params = config_params or {}
        self.frame_count = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.time_base = Fraction(1, 30)
        self.last_person_count = 0
        self.detection_area = None
        self.last_frame = None
        self.alert_manager = AlertManager()

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
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
               
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
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
           
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Resolución de cámara: {actual_width}x{actual_height}")
           
            # Leer frame de prueba
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("No se pudo leer frame de prueba")
               
            self.last_frame = frame
            logger.info("Cámara iniciada correctamente")
               
        except Exception as e:
            logger.error(f"Error iniciando cámara: {e}")
            if hasattr(self, 'cap'):
                self.cap.release()
            raise

    def update_config(self, new_config):
        try:
            if 'width' in new_config and 'height' in new_config:
                width = new_config['width']
                height = new_config['height']
               
                # Actualizar configuración de la cámara
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
               
                # Verificar si la resolución se aplicó correctamente
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
               
                logger.info(f"Resolución actualizada a: {actual_width}x{actual_height}")
               
                # Actualizar config_params
                self.config_params.update(new_config)
           
            if 'detection_area' in new_config:
                logger.info("\n=== RECIBIENDO NUEVA ÁREA DE DETECCIÓN ===")
                logger.info(f"Área recibida: {new_config['detection_area']}")
                self.detection_area = [int(x) for x in new_config['detection_area']]
                x1, y1, x2, y2 = self.detection_area
                logger.info(f"Área configurada: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
               
        except Exception as e:
            logger.error(f"Error actualizando configuración: {e}")
            raise

    async def recv(self):
        try:
            if not self.cap.isOpened():
                self.setup_camera()
               
            ret, img = self.cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                ret, img = self.cap.read()
                if not ret:
                    if self.last_frame is not None:
                        img = self.last_frame.copy()
                    else:
                        self.setup_camera()
                        ret, img = self.cap.read()
                        if not ret:
                            raise RuntimeError("Error persistente al leer frame de la cámara")
            else:
                self.last_frame = img.copy()

            current_time = time.time()
            if current_time - self.last_log_time >= 1.0:
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

            if self.detection_area is not None:
                x1_area, y1_area, x2_area, y2_area = self.detection_area
                cv2.rectangle(img, 
                            (x1_area, y1_area), 
                            (x2_area, y2_area), 
                            (0, 0, 255), 
                            3)

            boxes = results.boxes.xyxy
            people_in_area = 0
            max_confidence = 0.0
           
            if len(boxes) > 0:
                boxes = boxes.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
               
                person_indices = [i for i, cls in enumerate(class_ids) if cls == 0]
                current_person_count = len(person_indices)

                if person_indices:
                    detections = sv.Detections(
                        xyxy=boxes[person_indices],
                        confidence=confidences[person_indices],
                        class_id=class_ids[person_indices]
                    )
                   
                    for i, (box, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
                        x1, y1, x2, y2 = box.astype(int)
                        person_in_area = False

                        if self.detection_area is not None:
                            person_center_x = (x1 + x2) // 2
                            person_center_y = (y1 + y2) // 2
                           
                            if (x1_area <= person_center_x <= x2_area and 
                                y1_area <= person_center_y <= y2_area):
                                person_in_area = True
                                people_in_area += 1
                                max_confidence = max(max_confidence, conf)
                                logger.info("\n=== PERSONA DETECTADA EN ÁREA MARCADA ===")
                                cv2.circle(img, (person_center_x, person_center_y), 5, (0, 0, 255), -1)

                        rect_color = (0, 0, 255) if person_in_area else (0, 255, 0)
                        cv2.rectangle(img, (x1, y1), (x2, y2), rect_color, 2)
                       
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

                    if people_in_area > 0:
                        alert = DetectionAlert(
                            timestamp=time.time(),
                            person_count=people_in_area,
                            area_occupied=True,
                            confidence=float(max_confidence)
                        )
                        asyncio.create_task(self.alert_manager.send_alert(alert))
                    elif self.last_person_count > 0:
                        alert = DetectionAlert(
                            timestamp=time.time(),
                            person_count=0,
                            area_occupied=False,
                            confidence=1.0
                        )
                        asyncio.create_task(self.alert_manager.send_alert(alert))

                self.last_person_count = people_in_area

            await asyncio.sleep(0.001)

            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = self.frame_count
            new_frame.time_base = self.time_base
           
            self.frame_count += 1
            return new_frame
           
        except Exception as e:
            logger.error(f"Error en recv: {e}")
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