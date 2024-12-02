import os
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple

@dataclass
class CameraConfig:
    WIDTH: int = 640
    HEIGHT: int = 480
    FPS: int = 30
    BUFFER_SIZE: int = 1
    DEVICE_ID: int = 0

@dataclass
class SupervisionConfig:
    MODEL_PATH: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5
    DEVICE: str = "cuda"  # Cambiado de 'cpu' a 'cuda'
    PROCESS_EVERY_N_FRAMES: int = 1

@dataclass
class VisualizationConfig:
    SHOW_FPS: bool = True
    SHOW_CONFIDENCE: bool = True
    BOX_THICKNESS: int = 2
    BOX_COLOR: Tuple[int, int, int] = (0, 255, 0)  # BGR format
    TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)
    ENABLE_DEBUG_INFO: bool = True

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
    def create_supervision_config() -> SupervisionConfig:
        return SupervisionConfig()

    @staticmethod
    def create_visualization_config() -> VisualizationConfig:
        return VisualizationConfig()

    @staticmethod
    def create_server_config() -> ServerConfig:
        return ServerConfig()

    CAMERA: CameraConfig = field(default_factory=create_camera_config)
    SUPERVISION: SupervisionConfig = field(default_factory=create_supervision_config)
    VISUALIZATION: VisualizationConfig = field(default_factory=create_visualization_config)
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

        # Supervision config from env
        if os.getenv('SUPERVISION_MODEL_PATH'):
            config.SUPERVISION.MODEL_PATH = os.getenv('SUPERVISION_MODEL_PATH')
        if os.getenv('SUPERVISION_CONFIDENCE_THRESHOLD'):
            config.SUPERVISION.CONFIDENCE_THRESHOLD = float(os.getenv('SUPERVISION_CONFIDENCE_THRESHOLD'))
        if os.getenv('SUPERVISION_DEVICE'):
            config.SUPERVISION.DEVICE = os.getenv('SUPERVISION_DEVICE')
        if os.getenv('SUPERVISION_PROCESS_EVERY_N_FRAMES'):
            config.SUPERVISION.PROCESS_EVERY_N_FRAMES = int(os.getenv('SUPERVISION_PROCESS_EVERY_N_FRAMES'))
        if os.getenv('SUPERVISION_ENABLE_TRACKING'):
            config.SUPERVISION.ENABLE_TRACKING = os.getenv('SUPERVISION_ENABLE_TRACKING').lower() == 'true'

        # Visualization config from env
        if os.getenv('VISUALIZATION_SHOW_FPS'):
            config.VISUALIZATION.SHOW_FPS = os.getenv('VISUALIZATION_SHOW_FPS').lower() == 'true'
        if os.getenv('VISUALIZATION_SHOW_CONFIDENCE'):
            config.VISUALIZATION.SHOW_CONFIDENCE = os.getenv('VISUALIZATION_SHOW_CONFIDENCE').lower() == 'true'
        if os.getenv('VISUALIZATION_BOX_THICKNESS'):
            config.VISUALIZATION.BOX_THICKNESS = int(os.getenv('VISUALIZATION_BOX_THICKNESS'))
        if os.getenv('VISUALIZATION_DRAW_TRAJECTORIES'):
            config.VISUALIZATION.DRAW_TRAJECTORIES = os.getenv('VISUALIZATION_DRAW_TRAJECTORIES').lower() == 'true'

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