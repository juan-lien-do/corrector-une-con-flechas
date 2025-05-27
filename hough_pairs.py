# handdrawn_lines/hough_pairs.py
import cv2 as cv
import numpy as np

def hough_pairs(image_gray):
    """Detecta con HoughLinesP y devuelve lista raw_lines y una imagen con todos los pares."""
    # Binarizar
    _, bw = cv.threshold(image_gray, 220, 255, cv.THRESH_BINARY_INV)
    # Detectar segmentos
    raw = cv.HoughLinesP(bw, 1, np.pi/180, 20, minLineLength=20, maxLineGap=10)
    linesP = raw if raw is not None else []
    raw_lines = [tuple(l[0]) for l in linesP]
    # Dibujar para debug
    vis = cv.cvtColor(image_gray, cv.COLOR_GRAY2BGR)
    for x1,y1,x2,y2 in raw_lines:
        cv.line(vis, (x1,y1),(x2,y2),(0,0,255),1)
    return raw_lines, vis
