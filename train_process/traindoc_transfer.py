import json
import os

def save_docs_function(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        jsonfile = json.load(file)
    # # 图像输出路径
    # image_path = os.path.join(json_path, jsonfile['imagePath'])
    image_width = jsonfile['imageWidth']
    image_height = jsonfile['imageHeight']
    # txt文件输出路径
    txt_filename = os.path.splitext(os.path.basename(file_path))[0] + '.txt'
    output_txt_path = os.path.join(txt_save_path, txt_filename)
    # 储存关键点
    kpts = {'Mouth': [], 'Left pectoral fin': [], 'Right pectoral fin': [], 'Dorsal fin': [], 'Caudal fin': []}
    mouth = []
    mouth_v = 0
    mouth_flag = False
    l_fin = []
    l_fin_v = 0
    l_fin_flag = False
    r_fin = []
    r_fin_v = 0
    r_fin_flag = False
    dorsal = []
    dorsal_v = 0
    dorsal_fin_flag = False
    caudal = []
    caudal_v = 0
    caudal_fin_flag = False
    with open(output_txt_path, 'w') as txt:
        for shapes in jsonfile['shapes']:
            if shapes['label'] == 'Mouth':
                points = shapes['points']
                kpt_x = points[0][0] / image_width
                kpt_y = points[0][1] / image_height
                kpts['Mouth'].append([kpt_x,kpt_y])
            elif shapes['label'] == 'Left pectoral fin':
                points = shapes['points']
                kpt_x = points[0][0] / image_width
                kpt_y = points[0][1] / image_height
                kpts['Left pectoral fin'].append([kpt_x,kpt_y])
            elif shapes['label'] == 'Right pectoral fin':
                points = shapes['points']
                kpt_x = points[0][0] / image_width
                kpt_y = points[0][1] / image_height
                kpts['Right pectoral fin'].append([kpt_x,kpt_y])
            elif shapes['label'] == 'Dorsal fin':
                points = shapes['points']
                kpt_x = points[0][0] / image_width
                kpt_y = points[0][1] / image_height
                kpts['Dorsal fin'].append([kpt_x,kpt_y])
            elif shapes['label'] == 'Caudal fin':
                points = shapes['points']
                kpt_x = points[0][0] / image_width
                kpt_y = points[0][1] / image_height
                kpts['Caudal fin'].append([kpt_x,kpt_y])

        for shapes in jsonfile['shapes']:
            if shapes['label'] == 'myfish':
                class_id = category_list[shapes['label']]
                if class_id >= 0:
                    points = shapes['points']
                    x_min = min(points[0][0], points[1][0])
                    x_max = max(points[0][0], points[1][0])
                    y_min = min(points[0][1], points[1][1])
                    y_max = max(points[0][1], points[1][1])
                    # 归一化
                    x_center = (x_min + x_max) / 2 / image_width
                    y_center = (y_min + y_max) / 2 / image_height
                    width = (x_max - x_min) / image_width
                    height = (y_max - y_min) / image_height
                    # 找嘴巴
                    for m in kpts['Mouth']:
                        if (x_center - width/2) <= m[0] <= (x_center + width/2):
                            if (y_center - height/2) <= m[1] <= (y_center + height/2):
                                mouth = m
                                mouth_v = 2 # 可见且并未被遮挡
                                mouth_flag = True
                                break
                    if not mouth_flag:
                        mouth = [0, 0]
                        mouth_v = 0 # 不可见
                    # 找左侧鱼鳍
                    for l in kpts['Left pectoral fin']:
                        if (x_center - width/2) <= l[0] <= (x_center + width/2):
                            if (y_center - height/2) <= l[1] <= (y_center + height/2):
                                l_fin = l
                                l_fin_v = 2 # 可见且并未被遮挡
                                l_fin_flag = True
                                break
                    if not l_fin_flag:
                        l_fin = [0, 0]
                        l_fin_v = 0 # 不可见
                    # 找右侧鱼鳍
                    for r in kpts['Right pectoral fin']:
                        if (x_center - width/2) <= r[0] <= (x_center + width/2):
                            if (y_center - height/2) <= r[1] <= (y_center + height/2):
                                r_fin = r
                                r_fin_v = 2  # 可见且并未被遮挡
                                r_fin_flag = True
                                break
                    if not r_fin_flag:
                        r_fin = [0, 0]
                        r_fin_v = 0  # 不可见
                    # 找背鳍
                    for D in kpts['Dorsal fin']:
                        if (x_center - width/2) <= D[0] <= (x_center + width/2):
                            if (y_center - height/2) <= D[1] <= (y_center + height/2):
                                dorsal = D
                                dorsal_v = 2 # 可见且并未被遮挡
                                dorsal_fin_flag = True
                                break
                    if not dorsal_fin_flag:
                        dorsal = [0, 0]
                        dorsal_v = 0 # 不可见
                    # 找尾鳍
                    for c in kpts['Caudal fin']:
                        if (x_center - width/2) <= c[0] <= (x_center + width/2):
                            if (y_center - height/2) <= c[1] <= (y_center + height/2):
                                caudal = c
                                caudal_v = 2 # 可见且并未被遮挡
                                caudal_fin_flag = True
                                break
                    if not caudal_fin_flag:
                        caudal = [0, 0]
                        caudal_v = 0 # 不可见

                    txt.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {mouth[0]:.6f} {mouth[1]:.6f} {mouth_v} {l_fin[0]:.6f} {l_fin[1]:.6f} {l_fin_v} {r_fin[0]:.6f} {r_fin[1]:.6f} {r_fin_v} {dorsal[0]:.6f} {dorsal[1]:.6f} {dorsal_v} {caudal[0]:.6f} {caudal[1]:.6f} {caudal_v}\n")

json_path = r'D:\Myapps\anylabeling\fish\6'
txt_save_path = r'D:\AIModels\myfish_pose_Dataset\labels\test'
# image_save_path = r'D:\AIModels\ultralytics-main\myfish_Dataset\images\test'
# 识别类别
category_list = {'myfish': 0}

for filename in os.listdir(json_path):
    if filename.endswith('.json'):  # 处理.json文件
        print(filename)
        file_path = os.path.join(json_path, filename)
        save_docs_function(file_path)
