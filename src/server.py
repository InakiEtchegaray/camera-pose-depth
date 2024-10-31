import asyncio
import json
import logging
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
import os

from camera_processor import VideoTransformTrack

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set para mantener las conexiones activas
pcs = set()

class WebRTCServer:
    def __init__(self):
        """Inicializa el servidor WebRTC."""
        self.app = web.Application()
        self._init_routes()
        
    def _init_routes(self):
        """Inicializa las rutas del servidor."""
        # Obtener path del proyecto
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Configurar rutas para archivos estáticos
        self.app.router.add_static('/static', 
                                 os.path.join(project_root, "static"))
        
        # Configurar rutas principales
        self.app.router.add_get("/", self.index)
        self.app.router.add_post("/offer", self.offer)
        logger.info("Rutas del servidor inicializadas")

    async def offer(self, request):
        """Maneja las ofertas WebRTC."""
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()
        pcs.add(pc)
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Estado de conexión WebRTC: {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)
        
        try:
            video = VideoTransformTrack()
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
            logger.error(f"Error en el proceso de offer: {str(e)}")
            return web.Response(status=500, text=str(e))

    @staticmethod
    async def index(request):
        """Maneja la ruta principal y sirve la página web."""
        try:
            # Leer template
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            template_path = os.path.join(project_root, "templates", "index.html")
            
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            return web.Response(content_type="text/html", text=content)
        except Exception as e:
            logger.error(f"Error al servir la página principal: {str(e)}")
            return web.Response(status=500, text=str(e))

def run_server():
    """Inicia el servidor web."""
    try:
        server = WebRTCServer()
        print("\n=== Servidor de streaming con detección de poses y profundidad iniciado ===")
        print("Accede a http://localhost:8080 en tu navegador")
        print("Presiona Ctrl+C para detener el servidor\n")
        
        web.run_app(server.app, host="0.0.0.0", port=8080, access_log=None)
    except KeyboardInterrupt:
        print("\nServidor detenido por el usuario")
    finally:
        # Limpiar conexiones
        for pc in pcs:
            pc.close()

if __name__ == "__main__":
    run_server()