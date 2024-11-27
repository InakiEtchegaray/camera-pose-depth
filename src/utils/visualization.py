import cv2
import numpy as np

def draw_debug_info(frame, detections, fps):
    """Agrega información de debug al frame"""
    height, width = frame.shape[:2]
    
    # Crear un overlay semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (200, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # Agregar texto de información
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Personas: {len(detections)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame