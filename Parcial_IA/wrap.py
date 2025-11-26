import cv2
import numpy as np
from config import CARD_W, CARD_H
from utils import ordenar_puntos

def warp_carta(img, box):
    rect = ordenar_puntos(box)
    dst = np.array([
        [0, 0],
        [CARD_W-1, 0],
        [CARD_W-1, CARD_H-1],
        [0, CARD_H-1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (CARD_W, CARD_H))
    return warp

# si la carta está horizontal → girar
if warp.shape[0] < warp.shape[1]:
    warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)

# si está boca abajo → girar 180
upper_left = warp[5:40, 5:50]
blanco = np.sum(upper_left > 200)
negro = np.sum(upper_left < 50)

if negro > blanco:  
    warp = cv2.rotate(warp, cv2.ROTATE_180)

