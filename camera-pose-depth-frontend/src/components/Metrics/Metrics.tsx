import { IonCard, IonCardContent, IonGrid, IonRow, IonCol } from '@ionic/react';
import React, { useEffect, useState } from 'react';
import WebRTCService from '../../services/WebRTCService';
import './Metrics.css';

interface MetricsData {
  fps: number;
  status: string;
}

const Metrics: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricsData>({
    fps: 0,
    status: 'disconnected'
  });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    const interval = setInterval(async () => {
      try {
        if (mounted) {
          const data = await WebRTCService.getMetrics();
          setMetrics({
            fps: data?.fps || 0,
            status: data?.status || 'disconnected'
          });
          setError(null);
        }
      } catch (error) {
        if (mounted) {
          console.warn('Error fetching metrics:', error);
          setError('Error al obtener métricas');
        }
      }
    }, 1000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <IonCard className="metrics-card">
      <IonCardContent>
        {error ? (
          <div className="error-message">{error}</div>
        ) : (
          <IonGrid>
            <IonRow>
              <IonCol size="6">
                <div className="metric-item">
                  <div className="metric-label">FPS</div>
                  <div className="metric-value">
                    {metrics.fps.toFixed(1)}
                  </div>
                </div>
              </IonCol>
              <IonCol size="6">
                <div className="metric-item">
                  <div className="metric-label">Estado</div>
                  <div className="metric-value">
                    {metrics.status}
                  </div>
                </div>
              </IonCol>
            </IonRow>
          </IonGrid>
        )}
      </IonCardContent>
    </IonCard>
  );
};

export default Metrics;