import cv2
from PIL import Image

import RPi.GPIO as GPIO
import numpy as np
import matplotlib.pyplot as plt
import time
import os

import tflite_runtime.interpreter as tflite

# Converted trained model using TFLite converter
tflite_model = '/home/pi/Desktop/TFLite/converted_model.tflite'

# Initialize the video 
cam = cv2.VideoCapture(0)

# relay input to raspberry pi
relay_in = 16
status = 0

# set the relay as a output device
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(relay_in,GPIO.OUT)

frame_size = (480,640,3)

# color to extract from the frames
lower_hand = (0,0,70)
upper_hand = (255,255,255)

#print('Default frame size:',frame_size)

pred_digit = ''

# Initiate the TFLite Interpreter by allocating tensors
interpreter = tflite.Interpreter(model_path=tflite_model)
interpreter.resize_tensor_input(0,[1,224,224,3], strict=True)
interpreter.allocate_tensors()

# Index to get the input and output from the model
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

while True:
    pred_digit = ''
    ret, img = cam.read()
    if ret is False:
        continue
    
    img= cv2.flip(img,0)
    scene = img.copy()
    
    a, b, c, d = 340, 80, 300, 300
    cv2.rectangle(scene, (a, b), (a + c, b + d), (0, 255, 0), 2)
    cv2.imshow('Scene',scene)

    image = img[b:b + d, a:a + c]

    #blurred = cv2.GaussianBlur(image,(11,11),0)
    #cv2.imshow('Gaussian',blurred)

    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    #cv2.imshow('hsv',hsv)
    # mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    # Masking
    mask = cv2.inRange(hsv,lower_hand,upper_hand)
    #mask = cv2.erode(mask,kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
    #mask = cv2.dilate(mask, kernel, iterations=4)
    #res = cv2.bitwise_and(image, image, mask=mask)

    mask_3 = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    #cv2.imshow('Masked',mask)
    cv2.imshow('Mask 3 channels',mask_3)
    
    #cv2.imshow('bitwise and',res)
    #print('Mask shape:',mask.shape)
    #print('Mask 3 channel shape:',mask_3.shape)
    #print('bitwise and shape:',res.shape)
    
    _, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)
        
        for i in range(len(cnts)):
            # Create a mask image that contains the contour filled in
            cimg = np.zeros_like(image)
            cv2.drawContours(cimg, cnts, i, color=(255,255,255), thickness=-1)
            cv2.imshow('Contours masked',cimg)

        #print(cimg.shape)
        newImage = cv2.resize(cimg, (224, 224))
        newImage = np.array(newImage, dtype = np.float32)

        #input_data = newimage.reshape(224,224,3)
        input_data = np.expand_dims(newImage,axis=0)
        #print(input_data.shape)

        interpreter.set_tensor(input_index,input_data)
        interpreter.invoke()
        pred_digit = np.argmax(interpreter.get_tensor(output_index)) + 1 
        
        #print(pred_digit)
        print('digit predicted:',str(pred_digit))

    if pred_digit == 5:
        GPIO.output(relay_in,True)
        status = 1
    elif pred_digit == 1:
        GPIO.output(relay_in,False)
        status = 0
    
    if status == 1:
        cv2.putText(scene, "Light on", (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(scene, "Light off", (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #cv2.imshow('Masked',mask)
    cv2.imshow('Scene',scene)
    #cv2.imshow('Image',image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
