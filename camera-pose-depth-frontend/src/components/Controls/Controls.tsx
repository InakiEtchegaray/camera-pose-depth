import { IonCard, IonCardContent, IonItem, IonLabel, IonSelect, IonSelectOption, IonToggle } from '@ionic/react';
import React, { useEffect } from 'react';
import WebRTCService from '../../services/WebRTCService';
import './Controls.css';

interface Config {
  resolution: string;
  poseEnabled: boolean;
  depthEnabled: boolean;  // Cambiado de 'true' a 'boolean'
}

export const Controls: React.FC = () => {
  const [config, setConfig] = React.useState<Config>({
    resolution: '640,480',
    poseEnabled: true,
    depthEnabled: true
  });
  const [isUpdating, setIsUpdating] = React.useState(false);

  useEffect(() => {
    const savedConfig = localStorage.getItem('videoConfig');
    if (savedConfig) {
      try {
        const parsed = JSON.parse(savedConfig);
        setConfig(parsed);
      } catch (e) {
        console.error('Error loading saved config:', e);
      }
    }
  }, []);

  const handleConfigChange = async (newConfig: Partial<Config>) => {
    if (isUpdating) return;

    setIsUpdating(true);
    try {
      const updatedConfig = { ...config, ...newConfig };
      
      console.log('Updating config:', updatedConfig);
      
      await WebRTCService.updateConfig(updatedConfig);
      localStorage.setItem('videoConfig', JSON.stringify(updatedConfig));
      setConfig(updatedConfig);

      if (newConfig.resolution && newConfig.resolution !== config.resolution) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const videoElement = document.querySelector('video');
        if (videoElement) {
          await WebRTCService.disconnect();
          await new Promise(resolve => setTimeout(resolve, 500));
          await WebRTCService.setupConnection(videoElement as HTMLVideoElement);
        }
      }

    } catch (error) {
      console.error('Error updating config:', error);
    } finally {
      setIsUpdating(false);
    }
  };

  return (
    <IonCard className="controls-card">
      <IonCardContent>
        <IonItem>
          <IonLabel>Resolution</IonLabel>
          <IonSelect
            value={config.resolution}
            onIonChange={e => handleConfigChange({ resolution: e.detail.value })}
            disabled={isUpdating}
          >
            <IonSelectOption value="640,480">640 x 480</IonSelectOption>
            <IonSelectOption value="1280,720">1280 x 720</IonSelectOption>
            <IonSelectOption value="1920,1080">1920 x 1080</IonSelectOption>
          </IonSelect>
        </IonItem>

        <IonItem>
          <IonLabel>Pose Detection</IonLabel>
          <IonToggle
            checked={config.poseEnabled}
            onIonChange={e => handleConfigChange({ poseEnabled: e.detail.checked })}
            disabled={isUpdating}
          />
        </IonItem>

        <IonItem>
          <IonLabel>Depth Estimation</IonLabel>
          <IonToggle
            checked={config.depthEnabled}
            onIonChange={e => handleConfigChange({ depthEnabled: e.detail.checked })}
            disabled={isUpdating}
          />
        </IonItem>
      </IonCardContent>
    </IonCard>
  );
};

export default Controls;