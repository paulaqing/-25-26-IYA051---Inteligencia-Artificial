import cv2
import numpy as np
from config import HSV_LOWER, HSV_UPPER, AREA_MIN

def segmentar_carta(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(HSV_LOWER)
    upper = np.array(HSV_UPPER)
    mask = cv2.inRange(hsv, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask_clean = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask_clean

def encontrar_cartas(mask):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cartas = []
    for c in contornos:
        if cv2.contourArea(c) < AREA_MIN:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(c))
        cartas.append(np.int0(box))
    return cartas
