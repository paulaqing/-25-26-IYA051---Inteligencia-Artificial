#!/usr/bin/env python3
"""
main.py

Reconocimiento de cartas en tiempo real usando plantillas
de valores y palos creadas con captura_plantillas.py.
"""

import cv2
import numpy as np
import os

ANCHO_CARTA, ALTO_CARTA = 200, 300
AREA_MINIMA_CARTA = 2000
REGION_VALOR = (10, 20, 40, 50)
REGION_PALO  = (10, 70, 50, 50)

# ==== FUNCIONES ====

def cargar_plantillas(carpeta):
    plantillas = {}
    for nombre in os.listdir(carpeta):
        ruta = os.path.join(carpeta, nombre)
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            clave = os.path.splitext(nombre)[0]
            plantillas[clave] = img
    return plantillas

def segmentar_fondo_verde(imagen):
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    # ajusta estos valores si tu tapete es diferente
    verde_bajo = np.array([35, 60, 40])
    verde_alto = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, verde_bajo, verde_alto)
    mask_inv = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask_clean = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask_clean

def ordenar_puntos(caja):
    caja = np.array(caja, dtype="float32")
    suma = caja.sum(axis=1)
    diff = np.diff(caja, axis=1)
    rect = np.zeros((4,2), dtype="float32")
    rect[0] = caja[np.argmin(suma)]
    rect[2] = caja[np.argmax(suma)]
    rect[1] = caja[np.argmin(diff)]
    rect[3] = caja[np.argmax(diff)]
    return rect

def corregir_perspectiva(imagen, caja):
    puntos = ordenar_puntos(caja)
    destino = np.array([[0,0],[ANCHO_CARTA-1,0],[ANCHO_CARTA-1,ALTO_CARTA-1],[0,ALTO_CARTA-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(puntos, destino)
    warp = cv2.warpPerspective(imagen, M, (ANCHO_CARTA, ALTO_CARTA))
    return warp

def encontrar_carta(mask):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
    for c in contornos:
        if cv2.contourArea(c) < AREA_MINIMA_CARTA:
            return None
        rect = cv2.minAreaRect(c)
        caja = cv2.boxPoints(rect)
        return caja
    return None

def reconocer_plantilla(roi, plantillas):
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mejor_valor = None
    max_corr = -1
    for nombre, plantilla in plantillas.items():
        res = cv2.matchTemplate(roi_gray, plantilla, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > max_corr:
            max_corr = max_val
            mejor_valor = nombre
    return mejor_valor

# ==== PROGRAMA PRINCIPAL ====

def main():
    plantillas_valores = cargar_plantillas("plantillas/valores")
    plantillas_palos = cargar_plantillas("plantillas/palos")

    camara = cv2.VideoCapture(1)

    while True:
        ret, frame = camara.read()
        if not ret:
            break

        mask = segmentar_fondo_verde(frame)
        caja = encontrar_carta(mask)
        vista = frame.copy()

        if caja is not None:
            caja_int = np.array(caja, dtype=np.intp)
            cv2.drawContours(vista, [caja_int], -1, (0,255,0), 2)
            try:
                carta = corregir_perspectiva(frame, caja)
                x, y, w, h = REGION_VALOR
                roi_valor = carta[y:y+h, x:x+w]
                x2, y2, w2, h2 = REGION_PALO
                roi_palo = carta[y2:y2+h2, x2:x2+w2]

                valor = reconocer_plantilla(roi_valor, plantillas_valores)
                palo  = reconocer_plantilla(roi_palo, plantillas_palos)

                if valor and palo:
                    cv2.putText(vista, f"{valor} de {palo}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
            except:
                pass

        cv2.imshow("Reconocimiento de cartas", vista)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camara.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
