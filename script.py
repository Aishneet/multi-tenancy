import os
import pathlib
import torch
import numpy as np
import time
import cv2
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.dataset import read_label_file
from ultralytics import YOLO
from PIL import Image
from PIL import ImageDraw


labels = {0:'person', 1:'head'}
model = YOLO('objDet2_edgetpu.tflite',task='detect')
fer_model = 'fer1_edgetpu.tflite'
fer_labels = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'sadness', 6: 'surprise'}
interpreter = edgetpu.make_interpreter(fer_model)
interpreter.allocate_tensors()
size = common.input_size(interpreter)
params = common.input_details(interpreter, 'quantization_parameters')
scale = params['scales']
zero_point = params['zero_points']

image_dir="test_images/"
images= os.listdir(image_dir)
num_images = len(images)
total_time = 0
post_time = 0
obj_detec_inf_sp = 0
obj_detec_pre_sp = 0
obj_detec_pos_sp = 0
facial_inf = 0

for img_name in images:
    path = os.path.join(image_dir, img_name)
    image = cv2.imread(path)

    start_time = time.time()
    result = model.predict(image, imgsz=224)
    post_start = time.time()
    filtered_boxes = {}
    for r in result:
        boxes = r.boxes
        obj_detec_inf_sp += r.speed['inference']
        obj_detec_pre_sp += r.speed['preprocess']
        obj_detec_pos_sp += r.speed['postprocess']
        non_zero_indexes = torch.nonzero(boxes.cls == 1)
        filtered_boxes = boxes.xyxy[non_zero_indexes]
    for box in filtered_boxes:
        x1, y1, x2, y2 = box[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        face_image = image[y1:y2, x1:x2]
        try:
            resized_face_image = cv2.resize(face_image, size)
        except cv2.error as e:
            print("error: ", e)
            continue
        normalizedImg = (np.asarray(resized_face_image)-128)/(128*scale)+zero_point
        np.clip(normalizedImg, 0, 255, out = normalizedImg)
        common.set_input(interpreter, normalizedImg.astype(np.uint8))
        post_end = time.time()
        post_time+=(post_end-post_start)*1000
        fer_start = time.time()
        interpreter.invoke()
        fer_end = time.time()
        facial_inf += (fer_end -fer_start)*1000
        pr_st = time.time()
        classes = classify.get_classes(interpreter, top_k=1)
        print('%s: %.5f' % (fer_labels.get(classes[0].id, classes[0].id), classes[0].score))
        pr_ed = time.time()
        post_time += (pr_ed-pr_st)*1000
    end_time = time.time()
    total_time += (end_time -start_time)*1000
average_total_time = total_time/num_images
print("FER INF ", facial_inf)
print("OD PRE TIME: ", obj_detec_pre_sp)
print("OD POST TIME: ", obj_detec_pos_sp)
print("OD INFR TIME: ", obj_detec_inf_sp)
print("post_time: ", post_time)
print("TT:  ", total_time)
print("Average time: ", average_total_time)

