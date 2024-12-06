import React, { useEffect, useRef, useState } from 'react';
import './AreaSelector.css';

interface Point {
  x: number;
  y: number;
}

export interface AreaSelectorProps {
  videoElement: HTMLVideoElement | null;
  onAreaSelected: (area: { x1: number; y1: number; x2: number; y2: number }) => void;
}

const AreaSelector: React.FC<AreaSelectorProps> = ({ videoElement, onAreaSelected }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState<Point | null>(null);
  const [endPoint, setEndPoint] = useState<Point | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !videoElement) return;

    // Ajustar tamaño del canvas al video
    canvas.width = videoElement.clientWidth;
    canvas.height = videoElement.clientHeight;

    const drawArea = () => {
      const ctx = canvas.getContext('2d');
      if (!ctx || !startPoint) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      if (endPoint) {
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        
        const width = endPoint.x - startPoint.x;
        const height = endPoint.y - startPoint.y;
        
        ctx.strokeRect(startPoint.x, startPoint.y, width, height);
        ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
        ctx.fillRect(startPoint.x, startPoint.y, width, height);
      }
    };

    drawArea();
  }, [videoElement, startPoint, endPoint]);

  const convertToVideoCoordinates = (point: Point): Point => {
    if (!videoElement || !canvasRef.current) return point;

    const canvas = canvasRef.current;
    const videoRatio = {
      x: videoElement.videoWidth / canvas.width,
      y: videoElement.videoHeight / canvas.height
    };

    return {
      x: Math.round(point.x * videoRatio.x),
      y: Math.round(point.y * videoRatio.y)
    };
  };

  const getMousePos = (e: React.MouseEvent): Point => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDrawing(true);
    const point = getMousePos(e);
    console.log('Start Point (Canvas):', point);
    setStartPoint(point);
    setEndPoint(null);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing) return;
    const point = getMousePos(e);
    setEndPoint(point);
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
    if (startPoint && endPoint && videoElement) {
      // Convertir puntos del canvas a coordenadas del video
      const videoStart = convertToVideoCoordinates(startPoint);
      const videoEnd = convertToVideoCoordinates(endPoint);

      console.log('Start Point (Video):', videoStart);
      console.log('End Point (Video):', videoEnd);

      const area = {
        x1: Math.min(videoStart.x, videoEnd.x),
        y1: Math.min(videoStart.y, videoEnd.y),
        x2: Math.max(videoStart.x, videoEnd.x),
        y2: Math.max(videoStart.y, videoEnd.y)
      };

      console.log('Área calculada (Video):', area);
      onAreaSelected(area);
    } else {
      console.log('No hay puntos válidos:', { startPoint, endPoint });
    }
  };

  return (
    <div className="area-selector-container">
      <canvas
        ref={canvasRef}
        className="area-selector-canvas"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      />
    </div>
  );
};

export default AreaSelector;