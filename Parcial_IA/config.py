# config.py
import numpy as np

# Tamaño estándar del warp (ancho x alto)
CARD_W = 200
CARD_H = 300

# AREA mínima para considerar un contorno como carta
AREA_MIN = 1500

# Rango HSV del tapete (ajusta si tu tapete tiene otro tono)
HSV_LOWER = np.array([25, 40, 40])
HSV_UPPER = np.array([95, 255, 255])

# ROIs relativos al warp (x, y, w, h) — diseñados para CARD_W x CARD_H
# Valores ampliados para incluir "10" y símbolos grandes
ROI_VALOR = (10, 10, 41, 57)
ROI_PALO  = (5, 67, 44, 45)


# Carpetas
PLANTILLAS_VAL = "plantillas/valores"
PLANTILLAS_PAL = "plantillas/palos"
CAPTURAS_DIR   = "capturas"

# Plantilla tamaño normalizado (width x height) usado por reconocer
TPL_W = 60
TPL_H = 90

# Umbral mínimo de coincidencia (ajustable)
UMBRAL_MATCH = 0.38
