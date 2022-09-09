from  Infereclass import Inferer
import os
import os.path as osp
import math
from tqdm import tqdm
import cv2
import numpy as np
# import torch
from PIL import ImageFont
import time
from yolov6.utils.events import LOGGER, load_yaml

from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression

args  = {
    "weights": 'C:/Users/haharsha/Downloads/yolov6n.pt', # Path to weights file default weights are for nano model
    "source" : "C:/Users/haharsha/Documents/CAV/Camera_Flask_App-main/Camera_Flask_App-main/YOLOv6/data/images/image1.jpg", #Path to image file or it can be a directory of image
    "yaml"   : "C:/Users/haharsha/Documents/CAV/Camera_Flask_App-main/Camera_Flask_App-main/YOLOv6/data/coco.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.25, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "max-det" : 1000,  # maximal inferences per image
    "device" : 0,  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "save-img" : True,  # save visualized inference results.
    "classes" : None, # filter detection by classes
    "agnostic-nms": False,  # class-agnostic NMS
    "half" : False,   # whether to use FP16 half-precision inference.
    "hide-labels" : False,  # hide labels when saving visualization
    "hide-conf" : False # hide confidences.

}


def imagrec(frame):
    
    inferer = Inferer(weights = args['weights'], device = args['device'], yaml = args['yaml'], img_size = args['img-size'],half = args['half'], conf_thres= args['conf-thres'], iou_thres= args['iou-thres'],classes = args['classes'],
                  agnostic_nms = args['agnostic-nms'], max_det= args['max-det'])

    # try:
    #     # img_src = cv2.imread(frame)
    #     img_src = cv2.imread(args['source'])
    #     assert img_src is not None, f'Invalid image'
    # except Exception as e:
    #     print("Invalid Image Path or the image is empty cannot run inference")

    start = time.time()
    img, img_src = Inferer.process_image(frame, inferer.img_size, inferer.model.stride, args['half'])
    det = inferer.infer(img, img_src)
    end = time.time() - start
    fps_txt =  "{:.2f}".format(1/end)
    for *xyxy, conf, cls in reversed(det):
        class_num = int(cls)  # integer class
        label = None if args['hide-labels'] else (inferer.class_names[class_num] if args['hide-conf'] else f'{inferer.class_names[class_num]} {conf:.2f}')
        Inferer.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label, color=Inferer.generate_colors(class_num, True), fps = fps_txt)

    image = np.asarray(img_src)

    if args['save-img']:
        image_name = args['source'].split('/')[-1]
        cv2.imwrite(os.path.join("C:/Users/haharsha/Documents/CAV/Camera_Flask_App-main/Camera_Flask_App-main/YOLOv6/", image_name), image)

    return image
# cv2_imshow(image)