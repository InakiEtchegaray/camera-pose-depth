import { IonCard, IonCardContent, IonGrid, IonRow, IonCol } from '@ionic/react';
import React, { useEffect, useState } from 'react';
import WebRTCService from '../../services/WebRTCService';
import './Metrics.css';

interface Metrics {
  fps: number;
  cpu_usage: number;
  gpu_usage: number;
  latency: number;
}

const defaultMetrics: Metrics = {
  fps: 0,
  cpu_usage: 0,
  gpu_usage: 0,
  latency: 0
};

export const Metrics: React.FC = () => {
  const [metrics, setMetrics] = useState<Metrics>(defaultMetrics);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    const interval = setInterval(async () => {
      try {
        const newMetrics = await WebRTCService.getMetrics();
        if (mounted) {
          setMetrics(newMetrics);
          setError(null);
        }
      } catch (error) {
        console.warn('Error fetching metrics:', error);
        if (mounted) {
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
                  <div className="metric-value">{metrics.fps.toFixed(1)}</div>
                </div>
              </IonCol>
              <IonCol size="6">
                <div className="metric-item">
                  <div className="metric-label">Latencia</div>
                  <div className="metric-value">{metrics.latency.toFixed(0)} ms</div>
                </div>
              </IonCol>
            </IonRow>
            <IonRow>
              <IonCol size="6">
                <div className="metric-item">
                  <div className="metric-label">CPU</div>
                  <div className="metric-value">{metrics.cpu_usage.toFixed(1)}%</div>
                </div>
              </IonCol>
              <IonCol size="6">
                <div className="metric-item">
                  <div className="metric-label">GPU</div>
                  <div className="metric-value">{metrics.gpu_usage.toFixed(1)}%</div>
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