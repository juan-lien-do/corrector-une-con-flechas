# handdrawn_lines/classifier.py
import pandas as pd
import joblib
import tempfile, os
from math import atan2, degrees
import numpy as np

# Carga modelo entrenado
_model = joblib.load(os.path.join(os.path.dirname(__file__), "line_classifier.joblib"))

def _features(line1, line2):
    x1,y1,x2,y2 = line1; x3,y3,x4,y4 = line2
    # Características (repite lo mismo del dataset)
    angle1 = degrees(atan2(y2-y1, x2-x1)) % 180
    angle2 = degrees(atan2(y4-y3, x4-x3)) % 180
    diff_angle = abs(angle1-angle2)
    dist = min(np.hypot(x3-x1, y3-y1), np.hypot(x4-x1,y4-y1),
               np.hypot(x3-x2,y3-y2), np.hypot(x4-x2,y4-y2))
    shared = int(dist<5)
    len1 = np.hypot(x2-x1,y2-y1); len2 = np.hypot(x4-x3,y4-y3)
    rel_len = min(len1,len2)/max(len1,len2) if max(len1,len2)>0 else 0
    dot = np.dot([(x2-x1)/len1,(y2-y1)/len1],[(x4-x3)/len2,(y4-y3)/len2]) if len1*len2>0 else 0
    return [diff_angle, dist, shared, rel_len, dot]

def predict_pairs(raw_lines):
    """Dado raw_lines devuelve lista de pares a unir y una imagen debug."""
    # Construir DataFrame de pares
    rows = []
    for i, l1 in enumerate(raw_lines):
        for j, l2 in enumerate(raw_lines[i+1:], start=i+1):
            feat = _features(l1,l2)
            rows.append((i,j, *feat))
    df = pd.DataFrame(rows, columns=["i","j","diff_angle","min_endpoint_dist",
                                     "shared_endpoint","relative_length","dot_product"])
    # Predecir
    df["pred"] = _model.predict(df[["diff_angle","min_endpoint_dist",
                                     "shared_endpoint","relative_length","dot_product"]])
    # Imagen debug
    # (requiere pasar la imagen de fondo y dibujar solo pares con pred==1)
    return df[df["pred"]==1][["i","j"]].values
