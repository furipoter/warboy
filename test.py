import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from furiosa.runtime.sync import create_runner
from furiosa.quantizer import quantize, Calibrator, CalibrationMethod
import onnx
from utils.info import *
from utils.preprocess import *
from utils.postprocess import *
import time

image_path = './data/22_Picnic_Picnic_22_10.jpg'


with create_runner("yolov7_i8.onnx") as runner:
    image = cv2.imread(image_path)

    start = time.time()
    for i in range(30):
        image_tensor, preproc_params = preproc(image)
        output = runner.run(image_tensor)
        predictions = postproc(output, 0.65, 0.35)
        predictions = predictions[0]
        bboxed_img = draw_bbox(image, predictions, preproc_params)
        cv2.imwrite('./output.png', bboxed_img)
    print(str(time.time() - start))
