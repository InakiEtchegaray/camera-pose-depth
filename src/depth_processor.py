import torch
import logging
import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import time

logger = logging.getLogger(__name__)

class DepthProcessor:
    def __init__(self):
        """Inicializa el procesador de profundidad con configuración optimizada."""
        # Configuración de dispositivo y CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inicializando DepthProcessor en: {self.device}")
        
        # Optimizaciones CUDA
        if torch.cuda.is_available():
            logger.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = False

        self._init_model()
        self._init_cache()

    def _init_model(self):
        try:
            if torch.cuda.is_available():
                # Aumentar el límite de memoria GPU (ajusta el 0.8 según necesites, es 80% de la memoria)
                torch.cuda.set_per_process_memory_fraction(0.8)
                
                # Optimizaciones adicionales
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                self.depth_estimator = pipeline(
                    "depth-estimation",
                    model="LiheYoung/depth-anything-base-hf",
                    device=0,
                    torch_dtype=torch.float16,
                    model_kwargs={
                        "low_cpu_mem_usage": True,
                        "max_memory": {0: "6GiB"}  # Ajusta según tu GPU
                    }
                )
            else:
                # Configuración para CPU
                self.depth_estimator = pipeline(
                    "depth-estimation",
                    model="LiheYoung/depth-anything-base-hf",
                    device=-1,
                    torch_dtype=torch.float32
                )
            logger.info("Modelo de profundidad inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar modelo de profundidad: {str(e)}")
            raise

    def _init_cache(self):
        self.last_depth_map = None
        self.last_process_time = 0
        self.depth_process_interval = 0.3  # 200ms entre actualizaciones

    async def process_frame(self, frame):
        """
        Procesa un frame para obtener el mapa de profundidad.
        
        Args:
            frame (np.ndarray): Frame BGR de entrada
            
        Returns:
            np.ndarray: Frame con el mapa de profundidad superpuesto
        """
        current_time = time.time()
        
        if (current_time - self.last_process_time) >= self.depth_process_interval:
            try:
                # Reducir resolución para procesamiento
                depth_frame = cv2.resize(frame, (160, 120))
                image = Image.fromarray(cv2.cvtColor(depth_frame, cv2.COLOR_BGR2RGB))
                
                # Procesar con el modelo
                with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                    depth = self.depth_estimator(image)['predicted_depth']
                
                # Convertir a numpy y normalizar
                if torch.is_tensor(depth):
                    depth = depth.cpu()
                depth = depth.squeeze().numpy()
                depth = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
                
                # Crear mapa de color y redimensionar
                depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
                self.last_depth_map = cv2.resize(depth_color, 
                                               (frame.shape[1] // 4, frame.shape[0] // 4))
                
                self.last_process_time = current_time

                # Limpieza de memoria
                del depth
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error en procesamiento de profundidad: {str(e)}")

        # Superponer mapa de profundidad si existe
        if self.last_depth_map is not None:
            h, w = self.last_depth_map.shape[:2]
            frame[10:10+h, frame.shape[1]-w-10:frame.shape[1]-10] = self.last_depth_map
            
        return frame

    def __del__(self):
        """Limpieza de recursos."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error en limpieza de GPU: {str(e)}")