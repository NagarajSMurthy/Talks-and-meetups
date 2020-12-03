import numpy as np
import os
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
import time

# Converted trained model using TFLite converter
tflite_model = '/home/pi/Desktop/TFLite/converted_model.tflite'

# Test images location
test_imgs = '/home/pi/Desktop/TFLite/test/'

# To store the results
predictions = []
imgs = []
true_labels = []

# Initiate the TFLite Interpreter by allocating tensors
interpreter = tflite.Interpreter(model_path=tflite_model)
interpreter.resize_tensor_input(0,[1,224,224,3], strict=True)
interpreter.allocate_tensors()

# Index to get the input and output from the model
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

start_time = time.time()
for i in os.listdir(test_imgs):
    for img in os.listdir(test_imgs+str(i)):
        img = cv2.imread(test_imgs+str(i)+'/'+img)
        img = cv2.resize(img,(224,224))
        input_data = np.array(img, dtype = np.float32)
        input_data = np.expand_dims(input_data,axis=0)
        #print(input_data.shape)

        # Set the TFLite interpreter's input 
        interpreter.set_tensor(input_index,input_data)

        # Run the interpreter
        interpreter.invoke()

        # Get the predictions
        predictions.append(np.argmax(interpreter.get_tensor(output_index)))

        true_labels.append(i)
        imgs.append(img)
        
time_taken = time.time()-start_time

print('Completed inference on ',str(len(predictions)),' images')
print('Total time taken:',time_taken, 'seconds')

# Plot the test images along with the predictions
for j in range(len(predictions)):
    plt.imshow(imgs[j])
    print('Predicted:',predictions[j]+1)
    print('Actual:',true_labels[j])
    plt.show()
