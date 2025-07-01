import tensorflow as tf
import os
import sys
import numpy as np
import random
import math
import time
import cv2
from skimage.io import imread, imshow
from skimage.color import rgba2rgb
from tensorflow.keras.models import load_model
import utils as ut

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_PATH = 'test_case1.png'
MODEL_NAME = 'model_case1.keras'
POSE_GT = [1.16, 0.30, -1.18, 17.97, -3.09, -81.46]  # case1
# POSE_GT = [-0.12, -0.19, 0.13, 3.50, 4.92, -59.49]  # case6

# Load the test image
X = ut.imgPreprocess(imread(IMG_PATH))
X = X / 255.0
X = np.expand_dims(X, axis=0)
print(X.shape)

# Load the model
model = load_model(MODEL_NAME)

# Run the inference
start_time = time.time()
pose_pred = model.predict(X)
print("--- %s milliseconds ---" % (time.time() * 1000 - start_time * 1000))
print(pose_pred[0])

pose_err = ut.comparePoses(POSE_GT, pose_pred[0])
print('Pose Error: ' + str(pose_err))
