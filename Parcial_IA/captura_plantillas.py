#!/usr/bin/env python3
import cv2
import numpy as np
import os
from config import CARD_W, CARD_H, ROI_VALOR, ROI_PALO, PLANTILLAS_VAL, PLANTILLAS_PAL, CAPTURAS_DIR, TPL_W, TPL_H, AREA_MIN
from segmentacion import segmentar_fondo_verde
from warp import warp_carta
from utils import listar_camaras

os.makedirs(PLANTILLAS_VAL, exist_ok=True)
os.makedirs(PLANTILLAS_PAL, exist_ok=True)
os.makedirs(CAPTURAS_DIR, exist_ok=True)

def binarizar_para_plantilla(roi_color):
    gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.bitwise_not(th)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    th = cv2.resize(th, (TPL_W, TPL_H), interpolation=cv2.INTER_AREA)
    return th

def main():
    cams = listar_camaras()
    if not cams:
        print("No hay cámaras detectadas")
        return
    cam_index = cams[-1]
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    contador = 0
    print("Captura plantillas: 'c' carta completa, 'v' valor, 'p' palo, 'q' salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask = segmentar_fondo_verde(frame)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
            if cv2.contourArea(cnts[0]) > AREA_MIN:
                rect = cv2.minAreaRect(cnts[0])
                box = cv2.boxPoints(rect)
                warp = warp_carta(frame, box)

                xv, yv, wv, hv = ROI_VALOR
                xp, yp, wp, hp = ROI_PALO

                roi_val = warp[yv:yv+hv, xv:xv+wv]
                roi_pal = warp[yp:yp+hp, xp:xp+wp]

                # Dibujar rectángulos sobre warp para calibración
                cv2.rectangle(warp, (xv, yv), (xv+wv, yv+hv), (0,255,0), 2)  # Valor verde
                cv2.rectangle(warp, (xp, yp), (xp+wp, yp+hp), (255,0,0), 2)  # Palo azul

                cv2.imshow("Warp", warp)
                cv2.imshow("ROI Valor (presiona 'v')", cv2.resize(roi_val, (roi_val.shape[1]*2, roi_val.shape[0]*2)))
                cv2.imshow("ROI Palo (presiona 'p')", cv2.resize(roi_pal, (roi_pal.shape[1]*2, roi_pal.shape[0]*2)))

        cv2.imshow("Video", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and 'warp' in locals():
            ruta = os.path.join(CAPTURAS_DIR, f"carta_{contador:03d}.png")
            cv2.imwrite(ruta, warp)
            contador += 1
        elif key == ord('v') and 'roi_val' in locals():
            etiqueta = input("Nombre valor: ").strip()
            if etiqueta:
                binv = binarizar_para_plantilla(roi_val)
                ruta = os.path.join(PLANTILLAS_VAL, f"{etiqueta}.png")
                cv2.imwrite(ruta, binv)
        elif key == ord('p') and 'roi_pal' in locals():
            etiqueta = input("Nombre palo: ").strip()
            if etiqueta:
                binp = binarizar_para_plantilla(roi_pal)
                ruta = os.path.join(PLANTILLAS_PAL, f"{etiqueta}.png")
                cv2.imwrite(ruta, binp)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
