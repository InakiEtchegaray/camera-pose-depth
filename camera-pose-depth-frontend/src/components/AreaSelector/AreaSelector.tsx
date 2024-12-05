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
    console.log('Mouse Down triggered');
    setIsDrawing(true);
    const point = getMousePos(e);
    console.log('Start Point:', point);
    setStartPoint(point);
    setEndPoint(null);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing) return;
    const point = getMousePos(e);
    console.log('Mouse Move:', point);
    setEndPoint(point);
  };

  const handleMouseUp = () => {
    console.log('Mouse Up triggered');
    setIsDrawing(false);
    if (startPoint && endPoint) {
        console.log('Start Point:', startPoint);
        console.log('End Point:', endPoint);
        const area = {
            x1: Math.min(startPoint.x, endPoint.x),
            y1: Math.min(startPoint.y, endPoint.y),
            x2: Math.max(startPoint.x, endPoint.x),
            y2: Math.max(startPoint.y, endPoint.y)
        };
        console.log('Área calculada:', area);
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