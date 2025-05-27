import numpy as np
from typing import List, Tuple

def procesar_bounding_boxes(
    bb_blanco: List[Tuple[List[List[int]], str, float]],
    bb_resuelto: List[Tuple[List[List[int]], str, float]]
) -> List[Tuple[List[List[int]], str, float]]:
    """
    Procesa las bounding boxes del examen resuelto en base al examen en blanco.
    
    Args:
        bb_blanco: Lista de bounding boxes del examen en blanco (texto, posición)
        bb_resuelto: Lista de bounding boxes del examen resuelto
        
    Returns:
        Lista de bounding boxes procesadas, con las faltantes estimadas y las no deseadas eliminadas
    """
    # Extraer textos únicos del examen en blanco
    textos_blanco = {item[1] for item in bb_blanco}
    
    # Filtrar BB del resuelto: solo las que aparecen en el blanco
    bb_filtradas = [bb for bb in bb_resuelto if bb[1] in textos_blanco]
    
    # Identificar textos faltantes en el resuelto
    textos_resuelto = {bb[1] for bb in bb_filtradas}
    textos_faltantes = textos_blanco - textos_resuelto
    
    if not textos_faltantes:
        return bb_filtradas
    
    # Convertir a arrays numpy para cálculos
    def bb_to_array(bb_points):
        return np.array([point for point in bb_points], dtype=np.int32)
    
    # Crear diccionarios para acceso rápido
    blanco_por_texto = {bb[1]: bb_to_array(bb[0]) for bb in bb_blanco}
    resuelto_por_texto = {bb[1]: bb_to_array(bb[0]) for bb in bb_filtradas}
    
    # Para cada texto faltante, estimar su posición en el resuelto
    for texto in textos_faltantes:
        # Obtenemos la BB de referencia en el blanco
        bb_ref_blanco = blanco_por_texto[texto]
        
        # Encontrar BB vecinas en el blanco y sus correspondientes en el resuelto
        vecinos = []
        for t, bb in blanco_por_texto.items():
            if t in resuelto_por_texto:
                # Calcular desplazamiento relativo en el blanco
                centro_ref = np.mean(bb_ref_blanco, axis=0)
                centro_vecino = np.mean(bb, axis=0)
                desplazamiento = centro_vecino - centro_ref
                
                # Obtener la BB correspondiente en el resuelto
                bb_resuelto_vecino = resuelto_por_texto[t]
                centro_resuelto_vecino = np.mean(bb_resuelto_vecino, axis=0)
                
                # Calcular posición estimada en el resuelto
                centro_estimado = centro_resuelto_vecino - desplazamiento
                vecinos.append((desplazamiento, bb_resuelto_vecino))
        
        if not vecinos:
            continue  # No hay vecinos para estimar
        
        # Calcular BB estimada (promedio de las estimaciones de vecinos)
        nuevas_bb = []
        for desplazamiento, bb_vecino in vecinos:
            centro_vecino = np.mean(bb_vecino, axis=0)
            centro_estimado = centro_vecino - desplazamiento
            
            # Calcular dimensiones de la BB en blanco
            puntos_blanco = bb_ref_blanco
            min_x = min(p[0] for p in puntos_blanco)
            max_x = max(p[0] for p in puntos_blanco)
            min_y = min(p[1] for p in puntos_blanco)
            max_y = max(p[1] for p in puntos_blanco)
            
            ancho = max_x - min_x
            alto = max_y - min_y
            
            # Calcular nueva BB manteniendo las dimensiones originales
            x1 = int(centro_estimado[0] - ancho/2)
            y1 = int(centro_estimado[1] - alto/2)
            x2 = int(centro_estimado[0] + ancho/2)
            y2 = int(centro_estimado[1] + alto/2)
            
            nueva_bb = [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ]
            
            nuevas_bb.append(nueva_bb)
        
        # Promedio de todas las BB estimadas
        if nuevas_bb:
            # Convertir a array para calcular el promedio
            nuevas_bb_array = np.array(nuevas_bb, dtype=np.float32)
            bb_estimada_array = np.mean(nuevas_bb_array, axis=0).astype(np.int32).tolist()
            
            # Añadir con confianza baja ya que es estimada
            bb_filtradas.append((bb_estimada_array, texto, 0.5))
    
    return bb_filtradas