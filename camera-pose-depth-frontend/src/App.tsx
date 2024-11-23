import React, { useRef } from 'react';
import { IonApp, IonContent, IonPage, IonGrid, IonRow, IonCol } from '@ionic/react';
import { setupIonicReact } from '@ionic/react';

import '@ionic/react/css/core.css';
import '@ionic/react/css/normalize.css';
import '@ionic/react/css/structure.css';
import '@ionic/react/css/typography.css';
import '@ionic/react/css/padding.css';
import '@ionic/react/css/float-elements.css';
import '@ionic/react/css/text-alignment.css';
import '@ionic/react/css/text-transformation.css';
import '@ionic/react/css/flex-utils.css';
import '@ionic/react/css/display.css';

import VideoStream, { VideoStreamRef } from './components/VideoStream/VideoStream';
import Controls from './components/Controls/Controls';
import Metrics from './components/Metrics/Metrics';

setupIonicReact();

const App: React.FC = () => {
  const videoStreamRef = useRef<VideoStreamRef>(null);

  return (
    <IonApp>
      <IonPage>
        <IonContent>
          <IonGrid>
            <IonRow>
              <IonCol size="12" sizeMd="8">
                <VideoStream 
                  ref={videoStreamRef}
                />
              </IonCol>
              <IonCol size="12" sizeMd="4">
                <Controls />
                <Metrics />
              </IonCol>
            </IonRow>
          </IonGrid>
        </IonContent>
      </IonPage>
    </IonApp>
  );
};

export default App;