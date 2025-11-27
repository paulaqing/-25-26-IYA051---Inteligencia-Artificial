import numpy as np

CARD_W = 200
CARD_H = 300
AREA_MIN = 1500
HSV_LOWER = np.array([25, 40, 40])
HSV_UPPER = np.array([95, 255, 255])

# ROI exactos
ROI_VALOR = (7, 15, 50, 57)
ROI_PALO  = (7, 67, 44, 45)

PLANTILLAS_VAL = "plantillas/valores"
PLANTILLAS_PAL = "plantillas/palos"
CAPTURAS_DIR   = "capturas"

# Plantilla tamaño normalizado (para coincidencia)
TPL_W = 60
TPL_H = 90

UMBRAL_MATCH = 0.38
