#!/usr/bin/env python3
import cv2
import time
from config import CARD_W, CARD_H, AREA_MIN, ROI_VALOR, ROI_PALO, PLANTILLAS_VAL, PLANTILLAS_PAL, UMBRAL_MATCH
from segmentacion import segmentar_fondo_verde
from warp import warp_carta
from reconocimiento import cargar_plantillas, reconocer_roi

def main():
    plantillas_val = cargar_plantillas(PLANTILLAS_VAL)
    plantillas_pal = cargar_plantillas(PLANTILLAS_PAL)

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last = time.time()
    fps_cnt = 0
    print("Iniciando. Pulsa 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = segmentar_fondo_verde(frame)
        cv2.imshow("Mascara Verde", mask)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vista = frame.copy()

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
                cv2.rectangle(warp, (xv, yv), (xv+wv, yv+hv), (0,255,0), 2)
                cv2.rectangle(warp, (xp, yp), (xp+wp, yp+hp), (255,0,0), 2)
                cv2.imshow("Warp con ROIs", warp)

                # Reconocimiento
                val, val_score, tpl_val, top_vals = reconocer_roi(roi_val, plantillas_val, umbral=UMBRAL_MATCH, topn=3)
                pal, pal_score, tpl_pal, top_pals = reconocer_roi(roi_pal, plantillas_pal, umbral=UMBRAL_MATCH, topn=3)

                texto = f"{val or '???'} de {pal or '???'}"
                cv2.putText(vista, texto, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                cv2.putText(vista, f"V:{val_score:.2f} P:{pal_score:.2f}", (30,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                if tpl_val is not None:
                    cv2.imshow("Plantilla Valor", cv2.resize(tpl_val, (tpl_val.shape[1]*3, tpl_val.shape[0]*3)))
                if tpl_pal is not None:
                    cv2.imshow("Plantilla Palo", cv2.resize(tpl_pal, (tpl_pal.shape[1]*3, tpl_pal.shape[0]*3)))

        fps_cnt += 1
        now = time.time()
        if now - last >= 1.0:
            cv2.putText(vista, f"FPS: {fps_cnt}", (30,130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            last = now
            fps_cnt = 0

        cv2.imshow("Vista", vista)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
