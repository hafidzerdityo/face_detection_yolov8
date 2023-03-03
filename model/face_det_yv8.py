from ultralytics import YOLO
import ultralytics
import os 
import numpy as np
import time
from io import BytesIO
from PIL import Image
import base64
import cv2
from PIL import Image
import time

def current_milli_time():
    return time.time() * 1000

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
ultralytics.checks()

model = YOLO('./model/best.pt')

def convert_image_b64str(path):
    with open(path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read())
    img_str = img_b64.decode('ascii')
    return img_str

def predict_face_bansos(img_path: str) -> dict:
    dict_output = {
        'confidence': '',
        'result': ''
    }
    # dict_output['path'] = each_img_path
    results = model.predict(img_path, verbose=False)  

    try:
        dict_output['confidence'] = round(float(results[0].boxes.conf[0]),3)
    except:
        dict_output['confidence'] = float(0)

    try:
        if dict_output['confidence'] < 0.35:
            dict_output['result'] = False 
        else:
            dict_output['result'] = True 
    except:
        dict_output['result'] = False

    return dict_output

def predict_face_local_path(img_path: str) -> dict:
    dict_output = {
        'confidence': '',
        'result': '',
        'bboxes': ''
    }
    results = model.predict(img_path, conf=0.4)  
    try:
        dict_output['confidence'] = results[0].boxes.conf.tolist()
    except:
        dict_output['confidence'] = [0]
    boxes = results[0].boxes
    box = boxes  
    dict_output['bboxes'] = box.xyxy.tolist()

    if dict_output['bboxes']:
        dict_output['result'] = True 
    else:
        dict_output['result'] = False

    return dict_output

def predict_face_b64(img_b64str: str) -> dict:
    dict_output = {
        'bboxes': '',
        'confidence': '',
        'result': '',
        'image_predicted_b64': ''
    }
    orgimg = Image.open(BytesIO(base64.b64decode(img_b64str)))
    results = model.predict(orgimg, max_det=50, conf=0.4)  
    try:
        dict_output['confidence'] = results[0].boxes.conf.tolist()
    except:
        dict_output['confidence'] = [0]

    boxes = results[0].boxes
    box = boxes 
    dict_output['bboxes'] = box.xyxy.tolist()
    if dict_output['bboxes']:
        dict_output['result'] = True 
    else:
        dict_output['result'] = False
    
    if dict_output['result'] or dict_output['bboxes']:
        img = cv2.cvtColor(np.array(orgimg), cv2.COLOR_RGB2BGR)
        for each_box, each_conf in zip(dict_output['bboxes'], dict_output['confidence']):
            x1,y1,x2,y2 = each_box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (105,255,0), 2)
            cv2.putText(img,f"Face: {round(each_conf* 100,2)}%",(int(x1), int(y1) - 10),fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale = 0.6,color = (105,255,0),thickness=2)
        img = cv2.imencode('.jpg', img)[1].tobytes()
        b64_str_out = base64.b64encode(img).decode('ascii')
        dict_output['image_predicted_b64'] =  b64_str_out
        return {'b64_pred' :dict_output['image_predicted_b64'], 'result': dict_output['result']}
    else:
        return {'b64_pred': '', 'result' :dict_output['result']}