# Configuración general del proyecto

# Rango HSV del tapete verde (ajustable según iluminación)
HSV_LOWER = [25, 40, 40]
HSV_UPPER = [95, 255, 255]

# Tamaño de carta warp
CARD_W = 200
CARD_H = 300

# Área mínima de contorno para considerar carta
AREA_MIN = 2000

# Coordenadas ROI para valor y palo (relativas a CARD_W x CARD_H)
ROI_VALOR = (10, 20, 40, 50)  # x, y, w, h
ROI_PALO = (10, 70, 40, 50)   # x, y, w, h
