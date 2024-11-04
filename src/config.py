import os
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class CameraConfig:
    WIDTH: int = 640
    HEIGHT: int = 480
    FPS: int = 30
    BUFFER_SIZE: int = 1
    DEVICE_ID: int = 0

@dataclass
class DepthConfig:
    MODEL_NAME: str = "LiheYoung/depth-anything-base-hf"
    UPDATE_INTERVAL: float = 0.5  # Aumentado para reducir la frecuencia de procesamiento
    PROCESS_WIDTH: int = 128  # Reducido para menor carga
    PROCESS_HEIGHT: int = 96   # Reducido para menor carga
    GPU_MEMORY_FRACTION: float = 0.6
    ENABLE_CACHE: bool = True
    CACHE_SIZE: int = 16
    USE_HALF_PRECISION: bool = True  # Nuevo: usar FP16 para mejor rendimiento en GPU

@dataclass
class PoseConfig:
    MODEL_COMPLEXITY: int = 0  # Reducido a 0 para menor complejidad
    MIN_DETECTION_CONFIDENCE: float = 0.6
    MIN_TRACKING_CONFIDENCE: float = 0.6
    ENABLE_SMOOTHING: bool = True
    MAX_NUM_HANDS: int = 2
    PROCESS_EVERY_N_FRAMES: int = 2  # Nuevo: procesar cada N frames

@dataclass
class ServerConfig:
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    STATIC_FOLDER: str = "static"
    TEMPLATE_FOLDER: str = "templates"
    DEBUG: bool = True

@dataclass
class Config:
    @staticmethod
    def create_camera_config() -> CameraConfig:
        return CameraConfig()

    @staticmethod
    def create_depth_config() -> DepthConfig:
        return DepthConfig()

    @staticmethod
    def create_pose_config() -> PoseConfig:
        return PoseConfig()

    @staticmethod
    def create_server_config() -> ServerConfig:
        return ServerConfig()

    CAMERA: CameraConfig = field(default_factory=create_camera_config)
    DEPTH: DepthConfig = field(default_factory=create_depth_config)
    POSE: PoseConfig = field(default_factory=create_pose_config)
    SERVER: ServerConfig = field(default_factory=create_server_config)

    @classmethod
    def from_env(cls) -> 'Config':
        """Crea una configuración desde variables de entorno."""
        config = cls()

        # Camera config from env
        if os.getenv('CAMERA_WIDTH'):
            config.CAMERA.WIDTH = int(os.getenv('CAMERA_WIDTH'))
        if os.getenv('CAMERA_HEIGHT'):
            config.CAMERA.HEIGHT = int(os.getenv('CAMERA_HEIGHT'))
        if os.getenv('CAMERA_FPS'):
            config.CAMERA.FPS = int(os.getenv('CAMERA_FPS'))
        if os.getenv('CAMERA_BUFFER_SIZE'):
            config.CAMERA.BUFFER_SIZE = int(os.getenv('CAMERA_BUFFER_SIZE'))
        if os.getenv('CAMERA_DEVICE_ID'):
            config.CAMERA.DEVICE_ID = int(os.getenv('CAMERA_DEVICE_ID'))

        # Depth config from env
        if os.getenv('DEPTH_MODEL_NAME'):
            config.DEPTH.MODEL_NAME = os.getenv('DEPTH_MODEL_NAME')
        if os.getenv('DEPTH_UPDATE_INTERVAL'):
            config.DEPTH.UPDATE_INTERVAL = float(os.getenv('DEPTH_UPDATE_INTERVAL'))
        if os.getenv('DEPTH_GPU_MEMORY_FRACTION'):
            config.DEPTH.GPU_MEMORY_FRACTION = float(os.getenv('DEPTH_GPU_MEMORY_FRACTION'))

        # Pose config from env
        if os.getenv('POSE_MODEL_COMPLEXITY'):
            config.POSE.MODEL_COMPLEXITY = int(os.getenv('POSE_MODEL_COMPLEXITY'))
        if os.getenv('POSE_MIN_DETECTION_CONFIDENCE'):
            config.POSE.MIN_DETECTION_CONFIDENCE = float(os.getenv('POSE_MIN_DETECTION_CONFIDENCE'))
        if os.getenv('POSE_MIN_TRACKING_CONFIDENCE'):
            config.POSE.MIN_TRACKING_CONFIDENCE = float(os.getenv('POSE_MIN_TRACKING_CONFIDENCE'))

        # Server config from env
        if os.getenv('SERVER_HOST'):
            config.SERVER.HOST = os.getenv('SERVER_HOST')
        if os.getenv('SERVER_PORT'):
            config.SERVER.PORT = int(os.getenv('SERVER_PORT'))
        if os.getenv('SERVER_DEBUG'):
            config.SERVER.DEBUG = os.getenv('SERVER_DEBUG').lower() == 'true'

        return config

# Crear instancia de configuración
config = Config.from_env()