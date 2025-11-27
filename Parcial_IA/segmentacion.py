# segmentacion.py
import cv2
import numpy as np
from config import HSV_LOWER, HSV_UPPER

def segmentar_fondo_verde(frame, abrir_iter=2, cerrar_iter=2):
    """
    Devuelve máscara binaria donde la carta es 1 (blanco) y el fondo es 0 (negro)
    Más robusta frente a ruido e iluminación.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask_inv = cv2.bitwise_not(mask)

    # Morfología más agresiva para eliminar pequeños artefactos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    mask_clean = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel, iterations=abrir_iter)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=cerrar_iter)

    # Opcional: eliminar pequeñas áreas con contour filtering (se puede dejar)
    return mask_clean
