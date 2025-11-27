#!/usr/bin/env python3
import cv2
from config import ROI_VALOR, ROI_PALO, PLANTILLAS_VAL, PLANTILLAS_PAL, AREA_MIN, UMBRAL_MATCH
from segmentacion import segmentar_fondo_verde
from warp import warp_carta
from reconocimiento import cargar_plantillas, reconocer_roi
from utils import listar_camaras

def main():
    plantillas_val = cargar_plantillas(PLANTILLAS_VAL)
    plantillas_pal = cargar_plantillas(PLANTILLAS_PAL)

    cams = listar_camaras()
    if not cams:
        print("No hay cámaras detectadas")
        return
    cam_index = cams[-1]
    print(f"Usando cámara {cam_index} (probablemente tu móvil)")

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Iniciando. Pulsa 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = segmentar_fondo_verde(frame)
        vista = frame.copy()
        cv2.imshow("Mascara Verde", mask)

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

                warp_debug = warp.copy()
                cv2.rectangle(warp_debug, (xv, yv), (xv+wv, yv+hv), (0,255,0), 2)
                cv2.rectangle(warp_debug, (xp, yp), (xp+wp, yp+hp), (255,0,0), 2)
                cv2.imshow("Warp con ROIs", warp_debug)

                val, val_score, tpl_val, top_vals = reconocer_roi(roi_val, plantillas_val, UMBRAL_MATCH)
                pal, pal_score, tpl_pal, top_pals = reconocer_roi(roi_pal, plantillas_pal, UMBRAL_MATCH)

                cv2.imshow("ROI Valor", roi_val)
                if tpl_val is not None:
                    cv2.imshow("Plantilla Valor", tpl_val)
                cv2.imshow("ROI Palo", roi_pal)
                if tpl_pal is not None:
                    cv2.imshow("Plantilla Palo", tpl_pal)

                texto = f"{val or '???'} de {pal or '???'}"
                cv2.putText(vista, texto, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                cv2.putText(vista, f"V:{val_score:.2f} P:{pal_score:.2f}", (30,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow("Vista", vista)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
