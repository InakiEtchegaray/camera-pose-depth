import { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react';
import { IonCard, IonCardContent, IonButton } from '@ionic/react';
import WebRTCService from '../../services/WebRTCService';
import AreaSelector from '../AreaSelector/AreaSelector';
import './VideoStream.css';

interface Props {
  onStreamReady?: (videoElement: HTMLVideoElement) => void;
}

export interface VideoStreamRef {
  videoElement: HTMLVideoElement | null;
  reconnect: () => Promise<void>;
}

export const VideoStream = forwardRef<VideoStreamRef, Props>((props, ref) => {
  const { onStreamReady } = props;
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSelectingArea, setIsSelectingArea] = useState(false);
  const mountedRef = useRef(true);

  const setupStream = async () => {
    if (!videoRef.current || !mountedRef.current) return;
    
    try {
      setIsConnecting(true);
      setError(null);
      
      await WebRTCService.setupConnection(videoRef.current);
      
      if (mountedRef.current) {
        onStreamReady?.(videoRef.current);
      }
      
    } catch (error) {
      console.error('Failed to setup stream:', error);
      if (mountedRef.current) {
        setError(error instanceof Error ? error.message : 'Error desconocido');
      }
    } finally {
      if (mountedRef.current) {
        setIsConnecting(false);
      }
    }
  };

  const handleAreaSelected = async (area: { x1: number; y1: number; x2: number; y2: number }) => {
    try {
        console.log('Área seleccionada en frontend:', area);
        const areaArray = [
            Math.round(area.x1),
            Math.round(area.y1),
            Math.round(area.x2),
            Math.round(area.y2)
        ] as [number, number, number, number];
        
        console.log('Enviando al servidor:', { detection_area: areaArray, resolution: '640,480' });
        
        await WebRTCService.updateConfig({
            resolution: '640,480',
            detection_area: areaArray
        });
        console.log('Área enviada correctamente');
        setIsSelectingArea(false);
    } catch (error) {
        console.error('Error detallado al actualizar área:', error);
        setError('Error al configurar área de detección');
    }
  };
  useEffect(() => {
    mountedRef.current = true;
    setupStream();

    return () => {
      mountedRef.current = false;
      WebRTCService.disconnect();
    };
  }, []);

  useImperativeHandle(ref, () => ({
    videoElement: videoRef.current,
    reconnect: setupStream
  }));

  return (
    <IonCard className="video-card">
      <IonCardContent className="video-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="video-element"
        />
        {isSelectingArea && (
          <AreaSelector
            videoElement={videoRef.current}
            onAreaSelected={handleAreaSelected}
          />
        )}
        {isConnecting && (
          <div className="connection-overlay">
            Conectando...
          </div>
        )}
        {error && (
          <div className="error-overlay">
            Error: {error}
          </div>
        )}
        <div className="controls-overlay">
          <IonButton
            onClick={() => setIsSelectingArea(!isSelectingArea)}
            className="area-selector-button"
          >
            {isSelectingArea ? 'Cancelar' : 'Seleccionar Área'}
          </IonButton>
        </div>
      </IonCardContent>
    </IonCard>
  );
});

VideoStream.displayName = 'VideoStream';

export default VideoStream;