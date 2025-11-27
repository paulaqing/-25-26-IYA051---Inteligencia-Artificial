# utils.py
import cv2
import numpy as np

def ordenar_puntos(box):
    """
    Ordena 4 puntos en TL, TR, BR, BL
    box: array-like 4x2
    """
    pts = np.array(box, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4,2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def listar_camaras(max_index=6):
    disponibles = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            disponibles.append(i)
            cap.release()
    return disponibles
