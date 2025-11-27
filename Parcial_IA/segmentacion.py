import cv2
import numpy as np
from config import HSV_LOWER, HSV_UPPER

def segmentar_fondo_verde(frame, abrir_iter=1, cerrar_iter=1):
    """
    Devuelve máscara binaria donde la carta es 1 y el fondo 0.
    Más robusta frente a ruido e iluminación.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask_inv = cv2.bitwise_not(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask_clean = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel, iterations=abrir_iter)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=cerrar_iter)

    return mask_clean
