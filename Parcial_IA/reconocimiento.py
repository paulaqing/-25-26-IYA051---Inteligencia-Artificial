# reconocimiento.py
import cv2
import numpy as np
import os
from pathlib import Path
from config import TPL_W, TPL_H, UMBRAL_MATCH

def cargar_plantillas(carpeta):
    plantillas = {}
    if not os.path.exists(carpeta):
        return plantillas
    for f in sorted(os.listdir(carpeta)):
        ruta = os.path.join(carpeta, f)
        if not os.path.isfile(ruta):
            continue
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Normalizar a binario (si no está ya)
        _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        plantillas[Path(f).stem] = img_bin
    return plantillas

def binarizar_roi_color(roi_color):
    """
    Binariza ROI color a imagen B/N lista para match:
    - CLAHE para corregir sombras
    - blur
    - otsu
    - invertir (símbolo blanco)
    - limpiar morfológicamente
    - redimensionar a TPL_W x TPL_H
    """
    from config import TPL_W, TPL_H
    if roi_color is None or roi_color.size == 0:
        return None
    gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    gray = clahe.apply(gray)

    # Suavizado
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Otsu
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invertir (símbolo blanco)
    th = cv2.bitwise_not(th)

    # Limpieza
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Normalizar tamaño a plantilla
    th = cv2.resize(th, (TPL_W, TPL_H), interpolation=cv2.INTER_AREA)
    return th

def reconocer_roi(roi_color, plantillas, umbral=UMBRAL_MATCH, topn=3):
    roi_bin = binarizar_roi_color(roi_color)
    if roi_bin is None:
        return None, 0.0, None, []

    resultados = []
    for nombre, tpl in plantillas.items():
        tpl_r = cv2.resize(tpl, roi_bin.shape[::-1], interpolation=cv2.INTER_AREA)
        tpl_r_s = cv2.GaussianBlur(tpl_r, (3,3), 0)
        roi_s = cv2.GaussianBlur(roi_bin, (3,3), 0)

        res = cv2.matchTemplate(roi_s, tpl_r_s, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, _ = cv2.minMaxLoc(res)
        resultados.append((nombre, float(maxv), tpl_r))

    resultados.sort(key=lambda x: x[1], reverse=True)
    top = [(r[0], r[1]) for r in resultados[:topn]]

    if not resultados:
        return None, 0.0, None, []

    mejor_nombre, mejor_score, mejor_tpl = resultados[0]
    if mejor_score < umbral:
        return None, mejor_score, mejor_tpl, top

    return mejor_nombre, mejor_score, mejor_tpl, top
