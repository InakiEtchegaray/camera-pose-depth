import asyncio
import json
import logging
import os
from aiohttp import web
from aiohttp_cors import setup as cors_setup, ResourceOptions
from aiortc import RTCPeerConnection, RTCSessionDescription
from collections import deque

from camera_processor import VideoTransformTrack

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set para mantener las conexiones activas
pcs = set()

class WebRTCServer:
    def __init__(self):
        """Inicializa el servidor WebRTC."""
        self.app = web.Application()
        self.pcs = set()
        self.active_tracks = []
        
        # Configurar CORS
        cors = cors_setup(self.app, defaults={
            "*": ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*",
                max_age=3600
            )
        })
        
        self._init_routes()
        
        # Aplicar CORS a todas las rutas
        for route in list(self.app.router.routes()):
            cors.add(route)

    def _init_routes(self):
        """Inicializa las rutas del servidor."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Configurar rutas estáticas
        self.app.router.add_static('/static', 
                                 os.path.join(project_root, "static"))
        
        # Configurar rutas API
        self.app.router.add_get("/", self.index)
        self.app.router.add_post("/offer", self.offer)
        self.app.router.add_post("/update-config", self.update_config)
        self.app.router.add_get("/metrics", self.get_metrics)
        
        logger.info("Rutas del servidor inicializadas")

    async def cleanup_old_connections(self):
        """Limpia las conexiones antiguas."""
        for pc in self.pcs.copy():
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                await pc.close()
                self.pcs.discard(pc)

    async def offer(self, request: web.Request) -> web.Response:
        """Maneja las ofertas WebRTC."""
        try:
            await self.cleanup_old_connections()
            
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
                    if pc in self.pcs:
                        self.pcs.discard(pc)
            
            try:
                # Crear track de video
                video = VideoTransformTrack({
                    'width': width,
                    'height': height,
                    'pose_enabled': initial_config.get('poseEnabled', True),
                    'depth_enabled': initial_config.get('depthEnabled', True)
                })
                
                # Limpiar tracks antiguos
                self.active_tracks = [track for track in self.active_tracks 
                                    if track.readyState != "ended"]
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
                logger.error(f"Error en proceso de offer: {e}")
                if pc in self.pcs:
                    await pc.close()
                    self.pcs.discard(pc)
                raise

        except Exception as e:
            logger.error(f"Error general en offer: {e}")
            return web.Response(
                status=500,
                text=json.dumps({"error": str(e)}),
                content_type="application/json"
            )

    async def update_config(self, request: web.Request) -> web.Response:
        """Actualiza la configuración de procesamiento."""
        try:
            data = await request.json()
            logger.info(f"Recibida nueva configuración: {data}")

            width, height = map(int, data['resolution'].split(','))
            pose_enabled = data['poseEnabled']
            depth_enabled = data['depthEnabled']

            # Actualizar tracks activos
            self.active_tracks = [track for track in self.active_tracks 
                                if track.readyState != "ended"]
            
            success = True
            for track in self.active_tracks:
                try:
                    track.update_config({
                        'width': width,
                        'height': height,
                        'pose_enabled': pose_enabled,
                        'depth_enabled': depth_enabled
                    })
                except Exception as e:
                    logger.error(f"Error actualizando track: {e}")
                    success = False

            return web.Response(
                content_type="application/json",
                text=json.dumps({"success": success})
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

    async def get_metrics(self, request: web.Request) -> web.Response:
        """Endpoint para obtener métricas."""
        try:
            # Limpiar tracks inactivos
            self.active_tracks = [track for track in self.active_tracks 
                                if track.readyState != "ended"]
            
            metrics = {
                'fps': 0,
                'cpu_usage': 0,
                'gpu_usage': 0,
                'latency': 0
            }
            
            if self.active_tracks:
                track = self.active_tracks[0]
                track_metrics = track.get_performance_metrics()
                metrics.update(track_metrics)

            return web.Response(
                content_type="application/json",
                text=json.dumps(metrics),
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET',
                    'Access-Control-Allow-Headers': 'Content-Type'
                }
            )
        except Exception as e:
            logger.error(f"Error obteniendo métricas: {e}")
            return web.Response(
                status=500,
                text=json.dumps({"error": str(e)}),
                content_type="application/json"
            )

    async def index(self, request: web.Request) -> web.Response:
        """Sirve la página principal."""
        return web.Response(
            content_type="text/plain",
            text="WebRTC Server Running"
        )

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
        # Limpiar conexiones
        loop = asyncio.get_event_loop()
        for pc in server.pcs:
            loop.run_until_complete(pc.close())
        server.pcs.clear()

if __name__ == "__main__":
    run_server()