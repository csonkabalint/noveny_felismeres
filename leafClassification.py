import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import numpy as np

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)
img = cv2.imread("imageProcessTestData/C11_1.jpg")
print(img)
img_test = [img]
# np.empty(shape=1, dtype=float)
print(img_test)
bruh = np.random.random_sample(input_shape)
input_data = np.array(img_test, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
val_max = output_data[0].max()
print(val_max)
result = np.where(output_data[0] == val_max)
print(result)
print(type(output_data))