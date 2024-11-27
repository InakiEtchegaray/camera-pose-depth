import supervision as sv
import cv2
from ultralytics import YOLO

def setup_camera():
    # Inicializar la cámara USB (normalmente 0 es la webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara")
    return cap

def setup_detector():
    # Inicializar YOLO y las herramientas de anotación
    model = YOLO('yolov8n.pt')  # Modelo pequeño y rápido
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    return model, box_annotator

def main():
    cap = setup_camera()
    model, box_annotator = setup_detector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Realizar detección
        results = model(frame, device='cpu')[0]  # Usa 'cuda' si tienes GPU
        detections = sv.Detections.from_yolov8(results)
        
        # Filtrar solo personas (clase 0)
        mask = [cls == 0 for cls in detections.class_id]
        detections = detections[mask]
        
        # Anotar frame con las detecciones
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=[f"Person {i}" for i in range(len(detections))]
        )
        
        # Mostrar el resultado
        cv2.imshow("Detección en tiempo real", frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()