
# flujo:
# requisitos: instalar reqs, foto del une con flechas sin resolver, fotos de une con flechas resueltos.
 
# detectar caracteres originales en mapa vacio
import requests
import json
import easyocr
from api import detect_lines
import numpy
import cv2 as cv
from bounding_boxes import procesar_bounding_boxes
from merge_lines_boxes import asociar_simbolos_a_lineas

def main_func():
    reader = easyocr.Reader(['ja','en']) # this needs to run only once to load the model into memory
    base_scan = reader.readtext('baseimg.png', min_size=5, contrast_ths=0.1, text_threshold=0.5)
    
    print("\n============= RESULTADOS DEL ESCANEO DE TEXTO DE IMAGEN BASE ============ \n")
    for obj in base_scan:
        print(obj)

    
    print(len(base_scan))

    # hacer un mapita o algo asi
    # por cada imagen:
    ruta_imagen = 'problem.png'
    ruta_soluciones = 'solution' # aca van las screenshots
    # detectar lineas
    print("\n============= RESULTADOS DEL ESCANEO DE LINEAS DE PROBLEMA ============ \n")
    lineas_detectadas = detect_lines(ruta_imagen, ruta_soluciones)

    for i in lineas_detectadas:
        print(i)
    # detectar texto
    print("\n============= RESULTADOS DEL ESCANEO DE TEXTO DE PROBLEMA ============ \n")
    pre_text = reader.readtext(ruta_imagen, min_size=5, contrast_ths=0.1, text_threshold=0.8)

    boxes_procesadas = procesar_bounding_boxes(base_scan, pre_text)

    imagen_debug = cv.imread(ruta_imagen)
    for i in boxes_procesadas:
        print(i)
        cv.line(imagen_debug, i[0][0], i[0][2], (0, 0, 255), 3, cv.LINE_AA)
        cv.putText(imagen_debug, i[1], i[0][0], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
    cv.imwrite("solution\\texto_encontrado.png", imagen_debug)

    # juntar lineas y objetos
    print("\n============= RESULTADOS DEL ANALISIS DE TEXTO Y LINEAS ============ \n")

    asociados = asociar_simbolos_a_lineas(lineas_detectadas, boxes_procesadas)


    # calificar con LLM
    print("\n============= CALIFICACION DEL EXAMEN ============ \n")
    simbolos = "\n Simbolos del une-con-flechas = "
    for i in base_scan:
        simbolos += " [" + i[1] + "] "

    asociados_simplificado = "\nEstas son las lineas que dibujó el estudiante y los simbolos que conectan: "
    for i in asociados:
        asociados_simplificado+=  "[" + str(i[1]) + " - " + str(i[2]) + "]; "

    pregunta = "Eres un profesor que está corrigiendo un ejercicio de une-con-flechas de japonés de nivel N5. Te vamos a proveer los símbolos que hay que unir, y " \
    "los resultados de las flechas que obtuvo nuestro sistema de reconocimiento de flechas. Tienes que asignar una calificación a esta actividad," \
    "indicando el porcentaje de los puntos obtenibles. Ten en cuenta que puede haber más flechas reconocidas de las que debería haber."
    pregunta += simbolos
    pregunta += asociados_simplificado

    data = {
        "model":"qwen3:8b",
        "messages":[{"role":"user", "content":pregunta}],
        "stream":False
    }

    url = "http://localhost:11434/api/chat"
    response = requests.post(url, json=data)

    response_json = json.loads(response.text)

    respuesta_ia = response_json["message"]["content"]
    print(respuesta_ia)


    # output resultado


if __name__ == '__main__':
    main_func()