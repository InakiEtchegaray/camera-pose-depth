class WebRTCService {
    private peerConnection: RTCPeerConnection | null = null;
    private config = {
        resolution: '640,480',
        poseEnabled: true,
        depthEnabled: true
    };
    private isConnecting = false;

    async setupConnection(videoElement: HTMLVideoElement): Promise<void> {
        if (this.isConnecting) {
            console.log('Ya hay una conexión en proceso');
            return;
        }

        try {
            this.isConnecting = true;
            console.log('Iniciando configuración de WebRTC');
            
            if (this.peerConnection) {
                console.log('Cerrando conexión existente');
                this.disconnect();
            }

            if (videoElement.srcObject) {
                console.log('Limpiando stream anterior');
                const tracks = (videoElement.srcObject as MediaStream).getTracks();
                tracks.forEach(track => track.stop());
                videoElement.srcObject = null;
            }

            console.log('Creando nueva conexión RTCPeerConnection');
            this.peerConnection = new RTCPeerConnection({
                iceServers: []
            });

            this.peerConnection.addEventListener('track', (evt) => {
                console.log('Track recibido:', evt.track.kind);
                if (evt.track.kind === 'video') {
                    console.log('Asignando stream de video');
                    videoElement.srcObject = evt.streams[0];
                }
            });

            this.peerConnection.addEventListener('connectionstatechange', () => {
                console.log('Estado de conexión:', this.peerConnection?.connectionState);
            });

            this.peerConnection.addEventListener('iceconnectionstatechange', () => {
                console.log('Estado de ICE:', this.peerConnection?.iceConnectionState);
            });

            console.log('Creando oferta');
            await this.peerConnection.setLocalDescription(
                await this.peerConnection.createOffer({
                    offerToReceiveVideo: true
                })
            );

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
            console.log('Respuesta recibida del servidor', answer);

            await this.peerConnection.setRemoteDescription(answer);
            console.log('Descripción remota establecida');

        } catch (error) {
            console.error('Error en setupConnection:', error);
            throw error;
        } finally {
            this.isConnecting = false;
        }
    }

    async updateConfig(newConfig: typeof this.config): Promise<void> {
        try {
            console.log('Actualizando configuración:', newConfig);
            const response = await fetch('/update-config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(newConfig)
            });

            if (!response.ok) {
                throw new Error('Failed to update configuration');
            }
            
            this.config = newConfig;
            console.log('Configuración actualizada correctamente');

        } catch (error) {
            console.error('Error updating config:', error);
            throw error;
        }
    }

    async getMetrics(): Promise<any> {
        try {
            const response = await fetch('/metrics');
            if (!response.ok) {
                throw new Error('Failed to get metrics');
            }
            return response.json();
        } catch (error) {
            console.error('Error getting metrics:', error);
            return {
                fps: 0,
                cpu_usage: 0,
                gpu_usage: 0,
                latency: 0
            };
        }
    }

    async getSupportedResolutions(): Promise<any> {
        try {
            const response = await fetch('/supported-resolutions');
            if (!response.ok) {
                throw new Error('Failed to get supported resolutions');
            }
            return response.json();
        } catch (error) {
            console.error('Error getting supported resolutions:', error);
            return [
                { width: 640, height: 480 },
                { width: 1280, height: 720 }
            ];
        }
    }

    disconnect(): void {
        if (this.peerConnection) {
            try {
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