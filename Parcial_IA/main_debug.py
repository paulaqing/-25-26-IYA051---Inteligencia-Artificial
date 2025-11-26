#!/usr/bin/env python3
"""
main_debug.py

Programa principal de depuración y reconocimiento de cartas.
Usa segmentación por tapete verde, warp de carta y reconocimiento
con plantillas (reconocimiento.reconocer_roi).

Ajusta REGION_VALOR / REGION_PALO según tu warp si es necesario.
"""

import cv2
import numpy as np
import time
from reconocimiento import cargar_plantillas, reconocer_roi

# ---------------- CONFIGURACIÓN ----------------
ANCHO_CARTA, ALTO_CARTA = 200, 300
AREA_MINIMA_CARTA = 1500

REGION_VALOR = (10, 20, 40, 50)
REGION_PALO  = (10, 70, 40, 50)

VERDE_BAJO = np.array([25, 40, 40])
VERDE_ALTO = np.array([95, 255, 255])

CARPETA_VAL = "plantillas/valores"
CARPETA_PALO = "plantillas/palos"

UMBRAL_MATCH = 0.45

# ---------------- FUNCIONES AUXILIARES ----------------
def segmentar_fondo_verde(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, VERDE_BAJO, VERDE_ALTO)
    mask_inv = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_clean = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask_clean

def ordenar_puntos(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4,2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def corregir_orientacion(warp):
    if warp.shape[1] > warp.shape[0]:
        warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)

    esquina = warp[5:40, 5:40]
    if esquina.size == 0:
        return warp
    gris = cv2.cvtColor(esquina, cv2.COLOR_BGR2GRAY)
    blancos = np.sum(gris > 200)
    negros = np.sum(gris < 50)
    if negros > blancos:
        warp = cv2.rotate(warp, cv2.ROTATE_180)
    return warp

def warp_carta(frame, box):
    rect = ordenar_puntos(box)
    dst = np.array([[0,0],[ANCHO_CARTA-1,0],[ANCHO_CARTA-1,ALTO_CARTA-1],[0,ALTO_CARTA-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(frame, M, (ANCHO_CARTA, ALTO_CARTA))
    warp = corregir_orientacion(warp)
    return warp

# ---------------- MAIN ----------------
def main():
    plantillas_val = cargar_plantillas(CARPETA_VAL)
    plantillas_pal = cargar_plantillas(CARPETA_PALO)

    if not plantillas_val:
        print("No hay plantillas de valores en", CARPETA_VAL)
    if not plantillas_pal:
        print("No hay plantillas de palos en", CARPETA_PALO)

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_time = time.time()
    fps_counter = 0

    print("Iniciando... pulsa 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se recibió frame de la cámara.")
            break

        mask = segmentar_fondo_verde(frame)
        cv2.imshow("Mascara Verde", mask)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vista = frame.copy()

        if cnts:
            cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
            if cv2.contourArea(cnts[0]) > AREA_MINIMA_CARTA:
                rect = cv2.minAreaRect(cnts[0])
                box = cv2.boxPoints(rect)
                box_int = np.int32(box)
                cv2.drawContours(vista, [box_int], -1, (0,255,0), 2)

                try:
                    warp = warp_carta(frame, box)
                    cv2.imshow("Warp", warp)

                    xv, yv, wv, hv = REGION_VALOR
                    xp, yp, wp, hp = REGION_PALO

                    roi_val = warp[yv:yv+hv, xv:xv+wv]
                    roi_pal = warp[yp:yp+hp, xp:xp+wp]

                    if roi_val is None or roi_val.size == 0:
                        roi_val = None
                    if roi_pal is None or roi_pal.size == 0:
                        roi_pal = None

                    val, val_score, tpl_val, top_vals = reconocer_roi(roi_val, plantillas_val, umbral=UMBRAL_MATCH, topn=3)
                    pal, pal_score, tpl_pal, top_pals = reconocer_roi(roi_pal, plantillas_pal, umbral=UMBRAL_MATCH, topn=3)

                    texto = f"{val or '???'} de {pal or '???'}"
                    cv2.putText(vista, texto, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                    cv2.putText(vista, f"V:{val_score:.2f} P:{pal_score:.2f}", (30,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                    if tpl_val is not None:
                        try:
                            cv2.imshow("ROI Valor", cv2.resize(roi_val, (roi_val.shape[1]*3, roi_val.shape[0]*3)))
                            cv2.imshow("Plantilla Valor", cv2.resize(tpl_val, (tpl_val.shape[1]*3, tpl_val.shape[0]*3)))
                        except Exception:
                            pass
                    if tpl_pal is not None:
                        try:
                            cv2.imshow("ROI Palo", cv2.resize(roi_pal, (roi_pal.shape[1]*3, roi_pal.shape[0]*3)))
                            cv2.imshow("Plantilla Palo", cv2.resize(tpl_pal, (tpl_pal.shape[1]*3, tpl_pal.shape[0]*3)))
                        except Exception:
                            pass

                    print("Top3 valores:", top_vals)
                    print("Top3 palos  :", top_pals)

                except Exception as e:
                    print("Error en warp/recon:", e)

        fps_counter += 1
        now = time.time()
        if now - last_time >= 1.0:
            cv2.putText(vista, f"FPS: {fps_counter}", (30,130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            last_time = now
            fps_counter = 0

        cv2.imshow("Vista", vista)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
