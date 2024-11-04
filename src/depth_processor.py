import torch
import logging
import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import time
import hashlib

logger = logging.getLogger(__name__)

class DepthProcessor:
    def __init__(self):
        """Inicializa el procesador de profundidad con configuración optimizada para GPU."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inicializando DepthProcessor en: {self.device}")
        
        if torch.cuda.is_available():
            # Optimizaciones agresivas de CUDA
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Establecer memoria máxima de CUDA
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(0.9)  # Usar hasta 90% de la GPU
            logger.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memoria total GPU: {total_memory / 1024**2:.2f} MB")

        self._init_model()
        self._init_cache()

    def _init_model(self):
        """Inicializa el modelo de estimación de profundidad."""
        try:
            if torch.cuda.is_available():
                self.depth_estimator = pipeline(
                    "depth-estimation",
                    model="LiheYoung/depth-anything-base-hf",
                    device=0,
                    torch_dtype=torch.float16,  # Usar FP16 para mejor rendimiento
                    model_kwargs={
                        "low_cpu_mem_usage": True,
                    }
                )
                
                # Mover modelo explícitamente a GPU y optimizar
                if hasattr(self.depth_estimator, 'model'):
                    self.depth_estimator.model = self.depth_estimator.model.to('cuda')
                    
            else:
                self.depth_estimator = pipeline(
                    "depth-estimation",
                    model="LiheYoung/depth-anything-base-hf",
                    device=-1
                )
            logger.info("Modelo de profundidad inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar modelo de profundidad: {str(e)}")
            raise

    def _init_cache(self):
        """Inicializa el sistema de caché."""
        self.last_depth_map = None
        self.last_process_time = 0
        self.depth_process_interval = 0.5  # Intervalo en segundos
        self.cache = {}
        self.cache_max_size = 16

    def _get_frame_hash(self, frame):
        """Genera un hash para el frame."""
        frame_sample = frame[::10, ::10].astype(np.uint8)
        return hashlib.md5(frame_sample.tobytes()).hexdigest()

    def _process_depth_output(self, depth):
        """Procesa la salida del modelo de profundidad."""
        try:
            if torch.is_tensor(depth):
                if depth.device.type == 'cuda':
                    # Procesar todo en GPU
                    depth = depth.cpu()
                depth = depth.squeeze().numpy()
            
            depth = depth.squeeze()
            normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255)
            return normalized.astype(np.uint8)
        except Exception as e:
            logger.error(f"Error en process_depth_output: {str(e)}")
            return np.zeros((160, 120), dtype=np.uint8)

    async def process_frame(self, frame):
        """Procesa un frame para obtener el mapa de profundidad."""
        current_time = time.time()
        
        if (current_time - self.last_process_time) >= self.depth_process_interval:
            try:
                # Reducir resolución
                depth_frame = cv2.resize(frame, (160, 120))
                
                # Verificar caché
                frame_hash = self._get_frame_hash(depth_frame)
                if frame_hash in self.cache:
                    self.last_depth_map = self.cache[frame_hash]
                    return self._overlay_depth_map(frame)
                
                # Convertir a RGB y luego a PIL Image
                rgb_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # Inferencia con el modelo
                with torch.inference_mode():
                    if torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            depth = self.depth_estimator(pil_image)['predicted_depth']
                    else:
                        depth = self.depth_estimator(pil_image)['predicted_depth']
                
                # Procesar salida
                depth_map = self._process_depth_output(depth)
                depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
                
                # Redimensionar para overlay
                self.last_depth_map = cv2.resize(
                    depth_color, 
                    (frame.shape[1] // 4, frame.shape[0] // 4)
                )
                
                # Actualizar caché
                if len(self.cache) >= self.cache_max_size:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[frame_hash] = self.last_depth_map
                
                self.last_process_time = current_time
                
                # Limpiar memoria GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error en procesamiento de profundidad: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
        return self._overlay_depth_map(frame)

    def _overlay_depth_map(self, frame):
        """Superpone el mapa de profundidad en el frame."""
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