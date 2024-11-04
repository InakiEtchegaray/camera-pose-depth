import cv2
import mediapipe as mp
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

class PoseProcessor:
    def __init__(self):
        """Inicializa el detector de poses y manos."""
        logger.info("Inicializando detector de poses y manos")
        
        # Inicializar variables
        self.frame_count = 0
        self.last_pose_results = None
        self.last_hands_results = None
        
        # Inicializar MediaPipe
        self.mp = mp
        self.mp_drawing = self.mp.solutions.drawing_utils
        self.mp_pose = self.mp.solutions.pose
        self.mp_hands = self.mp.solutions.hands
        
        self._init_detectors()
        self._init_drawing_specs()

    def _init_detectors(self):
        """Inicializa los detectores con configuración optimizada."""
        try:
            # Configuración más ligera para pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # Usar modelo más ligero
                smooth_landmarks=True,
                min_detection_confidence=0.5,  # Reducir umbral
                min_tracking_confidence=0.5,
                enable_segmentation=False  # Desactivar segmentación
            )
            
            # Configuración más ligera para manos
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            logger.info("Detectores inicializados correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar detectores: {str(e)}")
            raise

    def _init_drawing_specs(self):
        """Inicializa las especificaciones de dibujo."""
        self.pose_styles = {
            'landmarks': self.mp_drawing.DrawingSpec(
                color=(0, 255, 0),  # Verde
                thickness=2,
                circle_radius=2
            ),
            'connections': self.mp_drawing.DrawingSpec(
                color=(255, 0, 0),  # Rojo
                thickness=2
            )
        }
        
        self.hand_styles = {
            'landmarks': self.mp_drawing.DrawingSpec(
                color=(0, 0, 255),  # Azul
                thickness=2,
                circle_radius=2
            ),
            'connections': self.mp_drawing.DrawingSpec(
                color=(255, 255, 0),  # Amarillo
                thickness=2
            )
        }

    def process_frame(self, frame):
        """Procesa un frame con optimizaciones."""
        try:
            self.frame_count += 1
            
            # Procesar cada 3 frames para reducir carga
            if self.frame_count % 3 != 0:
                return self._draw_cached_results(frame)
            
            # Reducir resolución más agresivamente
            process_frame = cv2.resize(frame, (160, 120))
            rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
            
            results = self._process_pose(rgb_frame)
            
            # Solo procesar manos si se detectó pose
            if results and results.pose_landmarks:
                hands_results = self._process_hands(rgb_frame)
            else:
                hands_results = None
            
            self.last_pose_results = results
            self.last_hands_results = hands_results
            
            return self._draw_detections(frame, results, hands_results)
            
        except Exception as e:
            logger.error(f"Error en procesamiento: {str(e)}")
            return frame

    def _process_pose(self, rgb_frame):
        """Procesa la detección de pose."""
        try:
            return self.pose.process(rgb_frame)
        except Exception as e:
            logger.error(f"Error en detección de pose: {str(e)}")
            return self.last_pose_results

    def _process_hands(self, rgb_frame):
        """Procesa la detección de manos."""
        try:
            return self.hands.process(rgb_frame)
        except Exception as e:
            logger.error(f"Error en detección de manos: {str(e)}")
            return self.last_hands_results

    def _draw_detections(self, frame, pose_results, hands_results):
        """Dibuja las detecciones en el frame."""
        output_frame = frame.copy()
        
        if pose_results and pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                output_frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.pose_styles['landmarks'],
                connection_drawing_spec=self.pose_styles['connections']
            )
        
        if hands_results and hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.hand_styles['landmarks'],
                    connection_drawing_spec=self.hand_styles['connections']
                )
        
        return output_frame

    def _draw_cached_results(self, frame):
        """Dibuja los últimos resultados guardados."""
        return self._draw_detections(frame, self.last_pose_results, self.last_hands_results)

    def __del__(self):
        """Limpieza de recursos."""
        try:
            if hasattr(self, 'pose'):
                self.pose.close()
            if hasattr(self, 'hands'):
                self.hands.close()
        except Exception as e:
            logger.error(f"Error al liberar recursos: {str(e)}")