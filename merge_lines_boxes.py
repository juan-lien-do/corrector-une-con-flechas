import numpy as np
from sklearn.cluster import DBSCAN

def asociar_simbolos_a_lineas(lineas, bounding_boxes, max_dist=50, eps=30):
    """
    Asocia símbolos a los extremos de las líneas usando clustering para manejar múltiples líneas cercanas.
    
    Args:
        lineas: Lista de tuplas (x1, y1, x2, y2) representando las líneas
        bounding_boxes: Lista de tuplas (puntos, texto, confianza) representando las BB
        max_dist: Distancia máxima para considerar asociación símbolo-extremo
        eps: Parámetro de distancia para el clustering DBSCAN
        
    Returns:
        Lista de tuplas (linea, simbolo_inicio, simbolo_fin) donde cada símbolo puede ser None
    """
    # Extraer centros de las bounding boxes
    centros = []
    textos = []
    for bb in bounding_boxes:
        puntos = np.array(bb[0], dtype=np.int32)
        centro = np.mean(puntos, axis=0)
        centros.append(centro)
        textos.append(bb[1])
    
    centros = np.array(centros)
    
    # Para cada línea, encontrar símbolos cercanos a sus extremos
    resultados = []
    
    for linea in lineas:
        x1, y1, x2, y2 = linea
        extremo1 = np.array([x1, y1])
        extremo2 = np.array([x2, y2])
        
        # Calcular distancias a ambos extremos
        distancias1 = np.linalg.norm(centros - extremo1, axis=1)
        distancias2 = np.linalg.norm(centros - extremo2, axis=1)
        
        # Encontrar índices de símbolos dentro de la distancia máxima
        idx_cercanos1 = np.where(distancias1 < max_dist)[0]
        idx_cercanos2 = np.where(distancias2 < max_dist)[0]
        
        # Procesar extremo 1 con clustering si hay múltiples candidatos
        simbolo1 = None
        if len(idx_cercanos1) > 0:
            if len(idx_cercanos1) == 1:
                # Solo un candidato, tomar el más cercano
                idx = idx_cercanos1[0]
                simbolo1 = textos[idx]
            else:
                # Aplicar clustering para seleccionar el mejor grupo
                coords = centros[idx_cercanos1]
                clustering = DBSCAN(eps=eps, min_samples=1).fit(coords)
                
                # Encontrar el cluster más cercano al extremo
                distancias_cluster = []
                for label in set(clustering.labels_):
                    mask = clustering.labels_ == label
                    centro_cluster = np.mean(coords[mask], axis=0)
                    dist = np.linalg.norm(centro_cluster - extremo1)
                    distancias_cluster.append((dist, label))
                
                # Tomar el cluster más cercano
                if distancias_cluster:
                    dist, best_label = min(distancias_cluster, key=lambda x: x[0])
                    mask = clustering.labels_ == best_label
                    idx = idx_cercanos1[mask]
                    
                    # Dentro del cluster, tomar el más cercano
                    distancias = distancias1[idx]
                    idx_best = idx[np.argmin(distancias)]
                    simbolo1 = textos[idx_best]
        
        # Procesar extremo 2 con clustering si hay múltiples candidatos
        simbolo2 = None
        if len(idx_cercanos2) > 0:
            if len(idx_cercanos2) == 1:
                # Solo un candidato, tomar el más cercano
                idx = idx_cercanos2[0]
                simbolo2 = textos[idx]
            else:
                # Aplicar clustering para seleccionar el mejor grupo
                coords = centros[idx_cercanos2]
                clustering = DBSCAN(eps=eps, min_samples=1).fit(coords)
                
                # Encontrar el cluster más cercano al extremo
                distancias_cluster = []
                for label in set(clustering.labels_):
                    mask = clustering.labels_ == label
                    centro_cluster = np.mean(coords[mask], axis=0)
                    dist = np.linalg.norm(centro_cluster - extremo2)
                    distancias_cluster.append((dist, label))
                
                # Tomar el cluster más cercano
                if distancias_cluster:
                    dist, best_label = min(distancias_cluster, key=lambda x: x[0])
                    mask = clustering.labels_ == best_label
                    idx = idx_cercanos2[mask]
                    
                    # Dentro del cluster, tomar el más cercano
                    distancias = distancias2[idx]
                    idx_best = idx[np.argmin(distancias)]
                    simbolo2 = textos[idx_best]
        
        resultados.append((linea, simbolo1, simbolo2))
    
    return resultados