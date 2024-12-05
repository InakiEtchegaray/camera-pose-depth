interface Config {
    resolution: string;
    detection_area?: [number, number, number, number];  // tupla opcional de 4 números
}

class WebRTCService {
    private peerConnection: RTCPeerConnection | null = null;
    private config: Config = {
        resolution: '640,480'
    };

    private rtcConfig = {
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
    };

    async setupConnection(videoElement: HTMLVideoElement): Promise<void> {
        try {
            // Si hay una conexión previa, cerrarla
            this.disconnect();
            
            console.log('Iniciando configuración de WebRTC');

            if (videoElement.srcObject) {
                console.log('Limpiando stream anterior');
                const tracks = (videoElement.srcObject as MediaStream).getTracks();
                tracks.forEach(track => track.stop());
                videoElement.srcObject = null;
            }

            console.log('Creando nueva conexión RTCPeerConnection');
            this.peerConnection = new RTCPeerConnection(this.rtcConfig);

            this.peerConnection.addEventListener('track', (evt) => {
                console.log('Track recibido:', evt.track.kind);
                if (evt.track.kind === 'video') {
                    console.log('Asignando stream de video');
                    const stream = new MediaStream([evt.track]);
                    videoElement.srcObject = stream;
                    videoElement.play().catch(e => console.error('Error reproduciendo video:', e));
                }
            });

            this.peerConnection.addEventListener('connectionstatechange', () => {
                console.log('Estado de conexión:', this.peerConnection?.connectionState);
            });

            this.peerConnection.addEventListener('iceconnectionstatechange', () => {
                console.log('Estado de ICE:', this.peerConnection?.iceConnectionState);
            });

            console.log('Creando oferta');
            const offer = await this.peerConnection.createOffer({
                offerToReceiveVideo: true
            });

            await this.peerConnection.setLocalDescription(offer);

            console.log('Enviando oferta al servidor');
            const response = await fetch('/offer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    sdp: this.peerConnection.localDescription?.sdp,
                    type: this.peerConnection.localDescription?.type,
                    config: this.config
                })
            });

            if (!response.ok) {
                throw new Error(`Error del servidor: ${response.status}`);
            }

            const answer = await response.json();
            console.log('Respuesta recibida del servidor');

            await this.peerConnection.setRemoteDescription(answer);
            console.log('Descripción remota establecida');

        } catch (error) {
            console.error('Error en setupConnection:', error);
            this.disconnect();
            throw error;
        }
    }

    async updateConfig(newConfig: Partial<Config>): Promise<void> {
        try {
            // Mantener la resolución actual cuando se actualiza el área de detección
            const configToSend = {
                resolution: this.config.resolution,  // Mantener la resolución actual
                ...newConfig  // Agregar los nuevos parámetros
            };
            
            console.log('Enviando configuración al servidor:', configToSend);
            const response = await fetch('/update-config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(configToSend)
            });
    
            if (!response.ok) {
                const errorData = await response.json();
                console.error('Error del servidor:', errorData);
                throw new Error(`Error del servidor: ${response.status} - ${errorData.error || 'Error desconocido'}`);
            }
            
            const data = await response.json();
            console.log('Respuesta del servidor:', data);
            
            this.config = { ...this.config, ...newConfig };
            console.log('Configuración actualizada correctamente');
    
        } catch (error) {
            console.error('Error completo al actualizar config:', error);
            throw error;
        }
    }
    async getMetrics(): Promise<any> {
        try {
            const response = await fetch('/metrics');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            return {
                fps: data.fps || 0,
                status: data.status || 'disconnected'
            };
        } catch (error) {
            console.warn('Error getting metrics:', error);
            return {
                fps: 0,
                status: 'error'
            };
        }
    }

    disconnect(): void {
        if (this.peerConnection) {
            try {
                const senders = this.peerConnection.getSenders();
                senders.forEach(sender => {
                    if (sender.track) {
                        sender.track.stop();
                    }
                });
                this.peerConnection.close();
            } catch (error) {
                console.error('Error closing connection:', error);
            } finally {
                this.peerConnection = null;
                console.log('Conexión cerrada');
            }
        }
    }
}

export default new WebRTCService();