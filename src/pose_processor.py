import cv2
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

class PoseProcessor:
    def __init__(self):
        """
        Inicializa el detector de poses y manos con configuración optimizada.
        """
        logger.info("Inicializando detector de poses y manos")
        
        # Inicializar componentes de MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        self._init_detectors()
        self._init_drawing_specs()

    def _init_detectors(self):
        """
        Inicializa los detectores de pose y manos con configuración optimizada.
        """
        # Configuración optimizada para pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # Modo video para mejor rendimiento
            model_complexity=1,       # Balance entre velocidad y precisión
            smooth_landmarks=True,    # Suavizado de landmarks
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Configuración optimizada para manos
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,          # Limitar a 1 mano para mejor rendimiento
            model_complexity=0,       # Modelo más ligero
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def _init_drawing_specs(self):
        """
        Inicializa las especificaciones de dibujo para poses y manos.
        """
        # Especificaciones para pose
        self.pose_landmark_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Verde
            thickness=2,
            circle_radius=2
        )
        self.pose_connection_spec = self.mp_drawing.DrawingSpec(
            color=(255, 0, 0),  # Rojo
            thickness=2
        )
        
        # Especificaciones para manos
        self.hand_landmark_spec = self.mp_drawing.DrawingSpec(
            color=(0, 0, 255),  # Azul
            thickness=2,
            circle_radius=2
        )
        self.hand_connection_spec = self.mp_drawing.DrawingSpec(
            color=(255, 255, 0),  # Amarillo
            thickness=2
        )

    def process_frame(self, frame):
        """
        Procesa un frame para detectar y dibujar poses y manos.
        
        Args:
            frame (np.ndarray): Frame BGR de entrada
            
        Returns:
            np.ndarray: Frame con las detecciones dibujadas
        """
        try:
            # Reducir tamaño para procesamiento
            process_frame = cv2.resize(frame, (320, 240))
            rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
            
            # Detectar pose y manos
            pose_results = self.pose.process(rgb_frame)
            hands_results = self.hands.process(rgb_frame)
            
            # Crear copia para dibujo
            output_frame = frame.copy()
            
            # Dibujar pose si se detecta
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.pose_landmark_spec,
                    connection_drawing_spec=self.pose_connection_spec
                )
            
            # Dibujar manos si se detectan
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        output_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.hand_landmark_spec,
                        connection_drawing_spec=self.hand_connection_spec
                    )
            
            return output_frame
            
        except Exception as e:
            logger.error(f"Error en procesamiento de pose/manos: {str(e)}")
            return frame  # Retornar frame original en caso de error

    def __del__(self):
        """
        Limpieza de recursos.
        """
        try:
            self.pose.close()
            self.hands.close()
        except Exception as e:
            logger.error(f"Error al liberar recursos: {str(e)}")