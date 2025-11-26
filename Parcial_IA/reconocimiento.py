#!/usr/bin/env python3
"""
reconocimiento.py

Versión mejorada con:
- Binarización robusta usando CLAHE
- Normalización de tamaño
- Coincidencia estable con matchTemplate
- Limpieza morfológica
- Suavizado previo a la correlación

Esto hace que el reconocimiento sea consistente y no dependa de la luz.
"""

import cv2
import numpy as np
import os
from pathlib import Path


# ============================================================
# CARGA DE PLANTILLAS
# ============================================================

def cargar_plantillas(carpeta):
    """
    Carga plantillas en blanco y negro desde una carpeta.
    Devuelve dict {nombre: imagen_binaria_uint8}
    """
    plantillas = {}
    if not os.path.exists(carpeta):
        print(f"[cargar_plantillas] Carpeta no existe: {carpeta}")
        return plantillas

    for f in sorted(os.listdir(carpeta)):
        ruta = os.path.join(carpeta, f)
        if not os.path.isfile(ruta):
            continue
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Normalizar a binario por si hay grises
        _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        nombre = Path(f).stem
        plantillas[nombre] = img_bin

    return plantillas


# ============================================================
# BINARIZACIÓN ROBUSTA
# ============================================================

def binarizar_roi_color(roi_color):
    """
    Binarización estable del ROI:
    - Convierte a gris
    - Aplica CLAHE (muy importante para sombras y luz desigual)
    - Suaviza
    - Binariza con Otsu
    - Invierte (símbolo blanco)
    - Normaliza tamaño (60x90)
    """

    if roi_color is None or roi_color.size == 0:
        return None

    # Gris
    gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

    # CLAHE → evita problemas de luz, sombras y reflejos
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

    # Normalizar tamaño — CRÍTICO para matchTemplate
    th = cv2.resize(th, (60, 90), interpolation=cv2.INTER_AREA)

    return th


# ============================================================
# RECONOCIMIENTO POR PLANTILLAS
# ============================================================

def reconocer_roi(roi_color, plantillas, umbral=0.38, topn=3, es_palo=False):
    """
    Reconoce una ROI comparando con las plantillas.
    Retorna:
      (nombre_mejor | None, score_mejor, plantilla_usada, lista_topN)
    """

    roi_bin = binarizar_roi_color(roi_color)
    if roi_bin is None:
        return None, 0.0, None, []

    resultados = []

    for nombre, tpl in plantillas.items():
        # Igualar tamaño de la plantilla
        tpl_r = cv2.resize(tpl, roi_bin.shape[::-1], interpolation=cv2.INTER_AREA)

        # Suavizado para igualar estilos
        tpl_r_s = cv2.GaussianBlur(tpl_r, (3,3), 0)
        roi_s = cv2.GaussianBlur(roi_bin, (3,3), 0)

        # match
        res = cv2.matchTemplate(roi_s, tpl_r_s, cv2.TM_CCOEFF_NORMED)
        _, maxval, _, _ = cv2.minMaxLoc(res)

        resultados.append((nombre, float(maxval), tpl_r))

    if not resultados:
        return None, 0.0, None, []

    # Ordenar por score
    resultados.sort(key=lambda x: x[1], reverse=True)
    top = [(r[0], r[1]) for r in resultados[:topn]]

    mejor_nombre, mejor_score, mejor_tpl = resultados[0]

    # Umbral mínimo
    if mejor_score < umbral:
        return None, mejor_score, mejor_tpl, top

    return mejor_nombre, mejor_score, mejor_tpl, top


# ============================================================
# OPCIONAL: comparación por contorno
# ============================================================

def comparar_contorno(roi_color, plantilla_bin):
    """
    Comparación por forma (opcional).
    No usada ahora, pero está implementada.
    """
    def main_contorno(bin_img):
        cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(cnts, key=cv2.contourArea) if cnts else None

    roi_bin = binarizar_roi_color(roi_color)
    if roi_bin is None:
        return 0.0

    c1 = main_contorno(roi_bin)
    c2 = main_contorno(plantilla_bin)
    if c1 is None or c2 is None:
        return 0.0

    val = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0)
    val = max(0.0, min(1.5, val))
    return 1.0 - (val / 1.5)
