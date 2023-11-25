import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from furiosa.runtime.sync import create_runner
from furiosa.quantizer import quantize, Calibrator, CalibrationMethod
import onnx
from utils.info import *
from utils.preprocess import preproc
from utils.postprocess import postproc

image_path = './data/22_Picnic_Picnic_22_140.jpg'


with create_runner("yolov7_i8.onnx") as runner:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = preproc(image)
    output = runner.run(image_tensor[0])
    predictions = postproc(output, 0.65, 0.35)

    import pdb; pdb.set_trace()