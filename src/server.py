import asyncio
import json
import logging
import os
import psutil
import time
import torch
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from collections import deque

from camera_processor import VideoTransformTrack

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self):
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.frames_processed = 0
        self.process = psutil.Process()
        self.start_time = time.time()
        self.last_metrics = {}
        self.metrics_lock = asyncio.Lock()

    async def update_metrics(self, new_metrics):
        async with self.metrics_lock:
            self.last_metrics.update(new_metrics)
            self.frames_processed += 1
            current_time = time.time()
            if 'fps' in new_metrics:
                self.fps_history.append(new_metrics['fps'])

    def get_metrics(self):
        try:
            # Calcular FPS como promedio móvil
            fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
            if self.last_metrics and 'fps' in self.last_metrics:
                fps = self.last_metrics['fps']  # Usar FPS del último frame si está disponible

            # Métricas del sistema
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent

            # Métricas de GPU
            gpu_metrics = self._get_gpu_metrics()

            # Calcular latencia
            latency = (time.time() - self.last_frame_time) * 1000

            metrics = {
                'fps': fps,
                'latency': latency,
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'processed_frames': self.frames_processed,
                'uptime': time.time() - self.start_time,
                'gpu_usage': gpu_metrics.get('gpu_usage', None),
                'gpu_memory': gpu_metrics.get('gpu_memory', None)
            }

            # Incluir métricas adicionales del procesamiento de video
            if self.last_metrics:
                for key, value in self.last_metrics.items():
                    if key not in metrics:
                        metrics[key] = value

            return metrics
        except Exception as e:
            logger.error(f"Error al obtener métricas: {e}")
            return {}

    def _get_gpu_metrics(self):
        metrics = {}
        if torch.cuda.is_available():
            try:
                # Uso de memoria GPU
                gpu_memory_allocated = torch.cuda.memory_allocated(0)
                gpu_memory_reserved = torch.cuda.memory_reserved(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory

                metrics['gpu_memory'] = (gpu_memory_allocated / total_memory) * 100
                metrics['gpu_usage'] = (gpu_memory_reserved / total_memory) * 100

                # Intentar obtener temperatura si es posible
                if hasattr(torch.cuda, 'get_device_temperature'):
                    metrics['gpu_temperature'] = torch.cuda.get_device_temperature(0)
            except Exception as e:
                logger.error(f"Error al obtener métricas de GPU: {e}")

        return metrics

class WebRTCServer:
    def __init__(self):
        """Inicializa el servidor WebRTC."""
        self.app = web.Application()
        self.pcs = set()
        self.active_tracks = []
        self.metrics_collector = MetricsCollector()
        self._init_routes()

    def _init_routes(self):
        """Inicializa las rutas del servidor."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.app.router.add_static('/static', 
                                 os.path.join(project_root, "static"))
        self.app.router.add_get("/", self.index)
        self.app.router.add_post("/offer", self.offer)
        self.app.router.add_post("/update-config", self.update_config)
        self.app.router.add_get("/metrics", self.get_metrics)
        self.app.router.add_get("/supported-resolutions", self.get_supported_resolutions)
        
        logger.info("Rutas del servidor inicializadas")

    async def get_supported_resolutions(self, request: web.Request) -> web.Response:
        """Endpoint para obtener las resoluciones soportadas."""
        try:
            if self.active_tracks:
                resolutions = self.active_tracks[0]._get_supported_resolutions()
                return web.json_response([
                    {"width": width, "height": height}
                    for width, height in resolutions
                ])
            return web.json_response([
                {"width": 640, "height": 480},
                {"width": 1280, "height": 720}
            ])
        except Exception as e:
            logger.error(f"Error al obtener resoluciones: {e}")
            return web.json_response([], status=500)

    async def get_metrics(self, request: web.Request) -> web.Response:
        """Endpoint para obtener métricas del sistema."""
        return web.json_response(self.metrics_collector.get_metrics())

    async def update_config(self, request: web.Request) -> web.Response:
        """Actualiza la configuración de procesamiento."""
        try:
            data = await request.json()
            logger.info(f"Recibida nueva configuración: {data}")

            width, height = map(int, data['resolution'].split(','))
            pose_enabled = data['poseEnabled']
            depth_enabled = data['depthEnabled']

            for track in self.active_tracks:
                await track.update_config({
                    'width': width,
                    'height': height,
                    'pose_enabled': pose_enabled,
                    'depth_enabled': depth_enabled
                })

            return web.Response(
                content_type="application/json",
                text=json.dumps({"success": True})
            )
        except Exception as e:
            logger.error(f"Error al actualizar configuración: {e}")
            return web.Response(
                content_type="application/json",
                text=json.dumps({
                    "success": False,
                    "error": str(e)
                }),
                status=400
            )

    async def offer(self, request: web.Request) -> web.Response:
        """Maneja las ofertas WebRTC."""
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        initial_config = params.get("config", {})
        width, height = map(int, initial_config.get('resolution', '640,480').split(','))
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Estado de conexión: {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)
        
        try:
            # Crear track de video con métricas
            video = VideoTransformTrack({
                'width': width,
                'height': height,
                'pose_enabled': initial_config.get('poseEnabled', True),
                'depth_enabled': initial_config.get('depthEnabled', True)
            })
            
            # Configurar callback para métricas
            async def on_metrics_update(metrics):
                await self.metrics_collector.update_metrics(metrics)

            video.on_metrics_update = on_metrics_update
            
            self.active_tracks.append(video)
            pc.addTrack(video)
            
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            return web.Response(
                content_type="application/json",
                text=json.dumps({
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type
                })
            )
        except Exception as e:
            logger.error(f"Error en offer: {e}")
            if pc in self.pcs:
                self.pcs.discard(pc)
            raise

    async def index(self, request: web.Request) -> web.Response:
        """Sirve la página principal."""
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            template_path = os.path.join(project_root, "templates", "index.html")
            
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            return web.Response(content_type="text/html", text=content)
        except Exception as e:
            logger.error(f"Error al servir index: {e}")
            raise

async def cleanup_connections(app):
    """Limpia las conexiones al cerrar el servidor."""
    pcs = app['pcs']
    for pc in pcs:
        await pc.close()
    pcs.clear()

def run_server():
    """Inicia el servidor web."""
    try:
        server = WebRTCServer()
        server.app['pcs'] = server.pcs
        server.app.on_shutdown.append(cleanup_connections)
        
        print("\n" + "="*60)
        print(" Camera Pose Depth Server")
        print(f" URL: http://localhost:8080")
        print(" Presiona Ctrl+C para detener")
        print("="*60 + "\n")
        
        web.run_app(server.app, host="0.0.0.0", port=8080)
    except KeyboardInterrupt:
        print("\nServidor detenido por el usuario")
    finally:
        loop = asyncio.get_event_loop()
        for pc in server.pcs:
            loop.run_until_complete(pc.close())
        server.pcs.clear()

if __name__ == "__main__":
    run_server()