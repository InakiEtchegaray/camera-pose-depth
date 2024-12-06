import React, { useEffect, useRef, useState } from 'react';
import WebRTCService from '../../services/WebRTCService';
import AreaSelector from '../AreaSelector/AreaSelector';
import DetectionAlerts from '../DetectionAlerts/DetectionAlerts';

const VideoDetection: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null);
  const [showAreaSelector, setShowAreaSelector] = useState(false);
  const [lastState, setLastState] = useState<boolean>(false);
  const [alerts, setAlerts] = useState<Array<{
    id: number;
    message: string;
    timestamp: number;
  }>>([]);

  useEffect(() => {
    const setupVideo = async () => {
      if (videoRef.current) {
        try {
          await WebRTCService.setupConnection(videoRef.current);
          setVideoElement(videoRef.current);
        } catch (error) {
          console.error('Error al configurar video:', error);
        }
      }
    };

    setupVideo();

    const unsubscribe = WebRTCService.onAlert((alert) => {
      if (alert.type === 'detection_alert') {
        // Solo crear alerta si el estado ha cambiado
        if (alert.areaOccupied !== lastState) {
          const newAlert = {
            id: Date.now(),
            message: alert.areaOccupied 
              ? `${alert.personCount} persona(s) entró al área restringida (Confianza: ${(alert.confidence * 100).toFixed(1)}%)`
              : 'La persona ha salido del área restringida',
            timestamp: alert.timestamp
          };
          setAlerts(prev => [newAlert, ...prev].slice(0, 5));
          setLastState(alert.areaOccupied);
        }
      }
    });

    return () => {
      WebRTCService.disconnect();
      unsubscribe();
    };
  }, [lastState]);

  const handleAreaSelected = async (area: { x1: number; y1: number; x2: number; y2: number }) => {
    try {
      await WebRTCService.updateConfig({
        detection_area: [area.x1, area.y1, area.x2, area.y2]
      });
      setShowAreaSelector(false);
      
      // Mensaje inicial después de seleccionar el área
      setAlerts(prev => [{
        id: Date.now(),
        message: 'Área de detección configurada. Monitoreando...',
        timestamp: Date.now() / 1000
      }, ...prev].slice(0, 5));
      
    } catch (error) {
      console.error('Error al actualizar área:', error);
    }
  };

  return (
    <div className="relative w-full max-w-4xl mx-auto">
      <div className="relative">
        <video
          ref={videoRef}
          className="w-full h-auto"
          autoPlay
          playsInline
        />
        {showAreaSelector && videoElement && (
          <AreaSelector
            videoElement={videoElement}
            onAreaSelected={handleAreaSelected}
          />
        )}
        <div className="absolute top-4 right-4 z-20">
          <button
            onClick={() => setShowAreaSelector(!showAreaSelector)}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
          >
            {showAreaSelector ? 'Cancelar Selección' : 'Seleccionar Área'}
          </button>
        </div>
      </div>
      <DetectionAlerts alerts={alerts} />
    </div>
  );
};

export default VideoDetection;