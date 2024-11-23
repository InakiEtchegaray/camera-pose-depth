import { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react';
import { IonCard, IonCardContent } from '@ionic/react';
import WebRTCService from '../../services/WebRTCService';
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

  const setupStream = async () => {
    if (isConnecting || !videoRef.current) return;
    
    try {
      setIsConnecting(true);
      setError(null);
      
      await WebRTCService.setupConnection(videoRef.current);
      onStreamReady?.(videoRef.current);
      
    } catch (error) {
      console.error('Failed to setup stream:', error);
      setError(error instanceof Error ? error.message : 'Error desconocido');
    } finally {
      setIsConnecting(false);
    }
  };

  useEffect(() => {
    setupStream();
  }, []);

  useImperativeHandle(ref, () => ({
    videoElement: videoRef.current,
    reconnect: setupStream
  }), [videoRef.current]);

  return (
    <IonCard className="video-card">
      <IonCardContent className="video-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="video-element"
        />
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
      </IonCardContent>
    </IonCard>
  );
});

VideoStream.displayName = 'VideoStream';

export default VideoStream;