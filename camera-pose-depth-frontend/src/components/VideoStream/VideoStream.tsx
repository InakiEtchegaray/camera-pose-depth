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