#!/usr/bin/env python3
"""
captura_plantillas.py

Captura vídeo desde la cámara (Mac o iPhone) y permite guardar plantillas de
valor y palo de cartas para reconocimiento clásico de cartas.

Teclas:
 - 'c' : guardar carta completa en 'capturas/'
 - 'v' : guardar valor en 'plantillas/valores/'
 - 'p' : guardar palo en 'plantillas/palos/'
 - 'q' : salir del programa
"""

import cv2
import numpy as np
import os

# ==== CONFIGURACIÓN BÁSICA ====
ANCHO_CARTA, ALTO_CARTA = 200, 300  # tamaño de la carta normalizada
LIMITE_INFERIOR_HSV = np.array([35, 60, 40])   # rango verde bajo
LIMITE_SUPERIOR_HSV = np.array([90, 255, 255]) # rango verde alto
AREA_MINIMA_CARTA = 2000

# Zonas aproximadas de valor y palo
REGION_VALOR = (10, 20, 40, 50)
REGION_PALO  = (10, 70, 40, 50)

# Crear carpetas si no existen
os.makedirs("plantillas/valores", exist_ok=True)
os.makedirs("plantillas/palos", exist_ok=True)
os.makedirs("capturas", exist_ok=True)

# ==== FUNCIONES ====

def listar_camaras(max_index=5):
    """Devuelve una lista de índices de cámaras disponibles."""
    disponibles = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            disponibles.append(i)
            cap.release()
    return disponibles

def segmentar_fondo_verde(imagen):
    """Crea una máscara donde se conserva todo lo que NO sea verde."""
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    mascara_verde = cv2.inRange(hsv, LIMITE_INFERIOR_HSV, LIMITE_SUPERIOR_HSV)
    mascara_no_verde = cv2.bitwise_not(mascara_verde)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mascara_limpia = cv2.morphologyEx(mascara_no_verde, cv2.MORPH_OPEN, kernel, iterations=1)
    return mascara_limpia

def ordenar_puntos(caja):
    """Ordena los 4 puntos de una caja en orden TL, TR, BR, BL."""
    caja = np.array(caja, dtype="float32")
    suma = caja.sum(axis=1)
    diferencia = np.diff(caja, axis=1)
    rect = np.zeros((4,2), dtype="float32")
    rect[0] = caja[np.argmin(suma)]      # superior izquierda
    rect[2] = caja[np.argmax(suma)]      # inferior derecha
    rect[1] = caja[np.argmin(diferencia)]# superior derecha
    rect[3] = caja[np.argmax(diferencia)]# inferior izquierda
    return rect

def corregir_perspectiva(imagen, caja):
    """Corrige la perspectiva de la carta para verla de frente."""
    puntos = ordenar_puntos(caja)
    destino = np.array([
        [0, 0],
        [ANCHO_CARTA - 1, 0],
        [ANCHO_CARTA - 1, ALTO_CARTA - 1],
        [0, ALTO_CARTA - 1]
    ], dtype="float32")
    matriz = cv2.getPerspectiveTransform(puntos, destino)
    carta_corregida = cv2.warpPerspective(imagen, matriz, (ANCHO_CARTA, ALTO_CARTA))
    return carta_corregida

def encontrar_carta(mascara):
    """Busca el contorno más grande (la carta principal)."""
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

def binarizar_imagen(imagen_gris):
    """Convierte a blanco y negro con umbral adaptativo."""
    binaria = cv2.adaptiveThreshold(
        imagen_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return binaria

def guardar_imagen(imagen, ruta):
    cv2.imwrite(ruta, imagen)
    print(f"✅ Imagen guardada: {ruta}")

# ==== PROGRAMA PRINCIPAL ====

def main():
    print("🔍 Buscando cámaras disponibles...")
    cams = listar_camaras()
    if not cams:
        print("❌ No se detectó ninguna cámara.")
        return

    cam_index = cams[-1]  # normalmente la última es la cámara del iPhone
    print(f"📸 Usando cámara con índice {cam_index}")

    camara = cv2.VideoCapture(cam_index)
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    contador = 0

    print("\n=== Captura de plantillas iniciada ===")
    print("Coloca una carta sobre el tapete verde y usa las teclas:")
    print("  'c' → guardar carta completa en 'capturas/'")
    print("  'v' → guardar valor en 'plantillas/valores/'")
    print("  'p' → guardar palo en 'plantillas/palos/'")
    print("  'q' → salir\n")

    while True:
        ret, fotograma = camara.read()
        if not ret:
            print("⚠️ No se pudo leer la cámara.")
            break

        mascara = segmentar_fondo_verde(fotograma)
        caja = encontrar_carta(mascara)

        vista = fotograma.copy()
        if caja is not None:
            caja_int = np.array(caja, dtype=np.intp)
            cv2.drawContours(vista, [caja_int], -1, (0,255,0), 2)
            try:
                carta = corregir_perspectiva(fotograma, caja)
                cv2.imshow("Carta normalizada", carta)

                # Extraer regiones
                x, y, w, h = REGION_VALOR
                roi_valor = carta[y:y+h, x:x+w]
                x2, y2, w2, h2 = REGION_PALO
                roi_palo = carta[y2:y2+h2, x2:x2+w2]

                # Mostrar ampliadas
                val_mostrar = cv2.resize(roi_valor, (int(w*2.5), int(h*2.5)))
                palo_mostrar = cv2.resize(roi_palo, (int(w2*2.5), int(h2*2.5)))
                cv2.imshow("Región VALOR ('v')", val_mostrar)
                cv2.imshow("Región PALO ('p')", palo_mostrar)
            except Exception as e:
                print("Error al corregir perspectiva:", e)
        else:
            cv2.destroyWindow("Carta normalizada")

        cv2.imshow("Vídeo (presiona 'q' para salir)", vista)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('q'):
            break
        elif tecla == ord('c') and caja is not None:
            guardar_imagen(carta, f"capturas/carta_{contador:03d}.png")
            contador += 1
        elif tecla == ord('v') and caja is not None:
            etiqueta = input("Nombre del valor (ej: As, 2, 3…): ").strip()
            if etiqueta:
                gris = cv2.cvtColor(roi_valor, cv2.COLOR_BGR2GRAY)
                binaria = binarizar_imagen(gris)
                ruta = os.path.join("plantillas/valores", f"{etiqueta}.png")
                guardar_imagen(binaria, ruta)
        elif tecla == ord('p') and caja is not None:
            etiqueta = input("Nombre del palo (ej: Corazones, Picas…): ").strip()
            if etiqueta:
                gris = cv2.cvtColor(roi_palo, cv2.COLOR_BGR2GRAY)
                binaria = binarizar_imagen(gris)
                ruta = os.path.join("plantillas/palos", f"{etiqueta}.png")
                guardar_imagen(binaria, ruta)

    camara.release()
    cv2.destroyAllWindows()
    print("\nFinalizado correctamente.\n")

if __name__ == "__main__":
    main()
