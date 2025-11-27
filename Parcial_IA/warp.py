import cv2
import numpy as np
from utils import ordenar_puntos
from config import CARD_W, CARD_H

def warp_carta(frame, box):
    """
    Aplica perspectiva y corrige orientación:
    - si queda horizontal gira 90º
    - si esquina superior izquierda oscura (boca abajo) gira 180º
    """
    rect = ordenar_puntos(box)
    dst = np.array([[0,0],[CARD_W-1,0],[CARD_W-1,CARD_H-1],[0,CARD_H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(frame, M, (CARD_W, CARD_H))

    # Corregir orientación
    if warp.shape[1] > warp.shape[0]:
        warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)

    # Revisar brillo en esquina superior izquierda
    try:
        esquina = warp[5:25, 5:25]  # más pequeña para evitar símbolos grandes
        gris = cv2.cvtColor(esquina, cv2.COLOR_BGR2GRAY)
        blancos = (gris > 200).sum()
        negros  = (gris < 50).sum()
        if negros > blancos:
            warp = cv2.rotate(warp, cv2.ROTATE_180)
    except Exception:
        pass

    return warp
