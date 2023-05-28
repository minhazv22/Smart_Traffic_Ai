import cv2
import numpy as np
import base64
import json
from urllib import request, parse

def main():
    """
        The goal is simple
        We send a RGB image to the server
        The server processes it, turns it into a grayscale image and returns the grayscale image
        We display both the client side RGB image and server side grayscale image together
    """
    
    # server URL
    localhost_url = 'http://127.0.0.1:5000/'

    # prepare webcam
    cap = cv2.VideoCapture(0)

    while True:
        # read frame from webcam
        ret, frame = cap.read()
        dim=(224,224)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        # servers can only be sent strings or bytes, so we need to convert the image to a string format
        # base64 is the widely used method in this regard        
        base64_img_string = base64.b64encode(frame).decode('utf-8')
        
        # this is the data we will send to the server
        params = {
            'img': base64_img_string,
            'height': frame.shape[0],
            'width': frame.shape[1],
            'channels': frame.shape[2]
        }
        
        # prepare the data to be sent with the request, it will be a POST request
        data = parse.urlencode(params).encode('ascii')
        req =  request.Request(localhost_url, data=data)
        
        # send the request to the server
        res = request.urlopen(req)
        
        # the server sends the grayscale image back
        res_bytes = res.readlines()[0]
        
         # the server response is in json format, but again, in json string, so we need to process that string to 
        # turn it properly into json

        prediction_json_string = res_bytes.decode('utf8').replace("'", '"') 
        prediction_json = json.loads(prediction_json_string)
        
        # server sent the grayscale image in the form of base64 string under the property named 'processed_img'
        prediction = prediction_json['prediction']
        
        convert the base64 string to bytes, then to flattened array, then reshape to actual grayscale 2D array
        decoded_processed_img_buffer = base64.b64decode(processed_img_string)    
        decoded_processed_img_flattened = np.frombuffer(decoded_processed_img_buffer, dtype=np.uint8)
        decoded_img = np.reshape(decoded_processed_img_flattened, (224, 224))
        
        # show the two frames
        cv2.imshow("Client frame", frame)
        cv2.imshow("Server processed frame", decoded_img)

        print("From client side >> ", prediction)
        
        # if user presses 'Q' from keyboard, the program ends
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
