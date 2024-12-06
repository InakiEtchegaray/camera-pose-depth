import React, { useEffect, useRef } from 'react';

interface Alert {
  id: number;
  message: string;
  timestamp: number;
}

interface DetectionAlertsProps {
  alerts: Alert[];
}

const DetectionAlerts: React.FC<DetectionAlertsProps> = ({ alerts }) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    // Crear el elemento de audio una sola vez
    if (!audioRef.current) {
      audioRef.current = new Audio();
      // Usar un sonido corto en base64 para evitar problemas de carga
      audioRef.current.src = 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdH2DAACEho2PdmNbbXeAAACCiZOadmZbaXJ8AACBipeeemhbaG56AACBiZijgGtaaW57AACChZSkhm9YaHJ+AACChJKmiXFXZ3GBAACBg4+li3VWZm+CAACAgYyji3hVZW6EAACAfomhkHtUZG2GAACAfYefkn5SYmuIAACAfISdlIFRYWmJAACAfYGblYRPYGeKAACAfYCYl4dOXmWMAACAgH+WmIlNXWSPAACAgH2UmYtLW2KRAACAgHySmY5KWmCTAACAfnqQmpFIWV+WAACAfXmOm5RHWVyYAACAfHeMm5ZFWFuaAACBfHWLm5lEV1mdAACAfHOJnJxDVlieAACAfXKInZ9CVleghYJ5cIYA';
    }

    // Si hay nuevas alertas, reproducir el sonido
    if (alerts.length > 0) {
      audioRef.current?.play().catch(e => console.warn('Error reproduciendo sonido:', e));
    }
  }, [alerts.length]);

  if (alerts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 max-w-md w-full space-y-2">
      {alerts.map((alert) => (
        <div
          key={alert.id}
          className={`
            p-4 rounded-lg shadow-lg
            ${alert.message.includes('entró')
              ? 'bg-red-100 border-l-4 border-red-500'
              : 'bg-green-100 border-l-4 border-green-500'}
          `}
        >
          <div className="flex flex-col">
            <p className={`text-base font-bold ${
              alert.message.includes('entró') ? 'text-red-800' : 'text-green-800'
            }`}>
              {alert.message.includes('entró') ? '¡Alerta de Ingreso!' : 'Alerta de Salida'}
            </p>
            <p className="text-sm mt-1">
              {alert.message}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              {new Date(alert.timestamp * 1000).toLocaleTimeString()}
            </p>
          </div>
        </div>
      ))}
    </div>
  );
};

export default DetectionAlerts;