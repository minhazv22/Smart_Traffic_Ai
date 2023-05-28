from flask import Flask, request, jsonify
from predict import predict
import base64
import numpy as np
import base64
import cv2
import pickle
from collections import Counter

model = open("traffic.pkl","rb")
clf2 = pickle.load(model)


from process_img import process_img

result_array = []
app = Flask(__name__)

"""
    Simple flask server
"""

@app.route('/', methods=['POST'])
def post_main():
    global result_array
    # the client sent the server 4 data, parse those data
    img_string_base64 = request.values['img']
    img_height = request.values['height']
    img_width = request.values['width']
    img_channels = request.values['channels']
    
    # decode the image string to bytes
    decoded_buffer = base64.b64decode(img_string_base64)
    
    # get the flattened image array from the bytes
    decoded_img_flattened = np.frombuffer(decoded_buffer, dtype=np.uint8)
    # reshape the flattened image, now we have the original client side image that the client sent to us
    decoded_img = np.reshape(decoded_img_flattened, (int(img_height), int(img_width), int(img_channels)))
    
    # this is where the processing of the image is done. the image can be sent as the input to a machine learning model here
    # and then the output can be obtained. we just convert it to grayscale here
    processed_features_array=process_img(decoded_img)

    prediction = predict(clf2,processed_features_array)
    prediction = int(prediction[0])

    result_array.append(prediction)

    if len(result_array) >= 30:
    	print("Last result >> ", result_array[-1])
    	result_array = []

    # print(type(prediction[0])) 
    print("From server side >> ", prediction)

    #processed_img = process_img(decoded_img)
    
    # # encode the grayscale image to base64 and send it back to client in the form of json data
    #processed_img_b64 = base64.b64encode(processed_img).decode('utf-8')
    return jsonify({"prediction": prediction})
    #return "hello"


@app.route('/', methods=['GET'])
def get_main():
    return "Hello, world"

if __name__ == "__main__":
    app.run()