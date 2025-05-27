# handdrawn_lines/api.py
import os, cv2 as cv
from hough_pairs import hough_pairs
from classifier import predict_pairs
from merger import merge_lines
import math

def detect_lines(image_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # 1) Leer y preprocesar
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite(os.path.join(out_dir,"step_1_gray.png"), gray)

    # 2) Hough + paso 2
    raw_lines, vis_hough = hough_pairs(gray)
    cv.imwrite(os.path.join(out_dir,"step_2_hough.png"), vis_hough)

    # 3) Predecir pares
    pairs = predict_pairs(raw_lines)
    vis3 = vis_hough.copy()
    for i,j in pairs:
        x1,y1,x2,y2 = raw_lines[i]
        x3,y3,x4,y4 = raw_lines[j]
        cv.line(vis3,(x1,y1),(x2,y2),(0,255,255),2)
        cv.line(vis3,(x3,y3),(x4,y4),(0,255,255),2)
    cv.imwrite(os.path.join(out_dir,"step_3_pairs.png"), vis3)

    # 4) Merge
    merged = merge_lines(raw_lines, pairs)

    # 5) get rid of ultra small lines
    avg_dist = 0
    for line in merged:
        distance = math.sqrt((line[2] - line[0])**2 + (line[3] - line[1]) **2)
        avg_dist += distance
        #print(line, distance)

    avg_dist /= len(merged)
    #print(avg_dist)

    new_merged = []
    for line in merged:
        distance = math.sqrt((line[2] - line[0])**2 + (line[3] - line[1]) **2)
        if(distance > avg_dist * 0.6):
            new_merged.append(line)
    
    merged = new_merged

    vis4 = img.copy()
    for x1,y1,x2,y2 in merged:
        cv.line(vis4,(x1,y1),(x2,y2),(0,255,0),3)
    cv.imwrite(os.path.join(out_dir,"step_4_merged.png"), vis4)

    #print(merged)

    return merged


if __name__ == '__main__':
    detect_lines("problem.png", "solution")