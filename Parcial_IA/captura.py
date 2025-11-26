import cv2

def inicializar_camara(indice=0, ancho=1280, alto=720):
    cap = cv2.VideoCapture(indice)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    return cap

def capturar_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def liberar_camara(cap):
    cap.release()
    cv2.destroyAllWindows()
