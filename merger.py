# handdrawn_lines/merger.py
import numpy as np
from collections import defaultdict
import cv2 as cv

def merge_lines(raw_lines, pairs):
    """Agrupa por componentes conexos y calcula línea unificada por cluster."""
    g = defaultdict(set)
    for i,j in pairs:
        g[i].add(j); g[j].add(i)
    # Extraer clusters
    seen=set(); clusters=[]
    for n in g:
        if n not in seen:
            stack=[n]; cl=[]
            while stack:
                x=stack.pop()
                if x not in seen:
                    seen.add(x); cl.append(x)
                    stack.extend(g[x])
            clusters.append(cl)
    # Merge por PCA
    merged=[]
    for cl in clusters:
        pts=[]
        for idx in cl:
            x1,y1,x2,y2 = raw_lines[idx]
            pts.append((x1,y1)); pts.append((x2,y2))
        data=np.array(pts, dtype=np.float32)
        m, eig = cv.PCACompute(data, mean=np.array([]))
        vx,vy = eig[0]; x0,y0 = m[0]
        # proyección
        ts=[]
        for x,y in pts:
            t = (vx*(x-x0)+vy*(y-y0))/(vx*vx+vy*vy)
            ts.append((t,(x0+t*vx,y0+t*vy)))
        ts.sort(key=lambda z:z[0])
        (x1,y1),(x2,y2) = ts[0][1], ts[-1][1]
        merged.append((int(x1),int(y1),int(x2),int(y2)))
    return merged
