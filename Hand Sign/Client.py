import socket

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)
from keras.models import load_model
from collections import deque
import mediapipe as mp
from src.utils import get_images, get_overlay
from src.config import *
import matplotlib.pylab as plt
from tensorflow.keras.preprocessing.image import img_to_array
detector = HandDetector(maxHands = 1)
classifier = Classifier("D:\Git do an\Do-An-TKLL\Hand Sign\Model_AirGesture_Sign\keras_model.h5",
                        "D:\Git do an\Do-An-TKLL\Hand Sign\Model_AirGesture_Sign\labels.txt")

model = load_model("D:\Git do an\Do-An-TKLL\Hand Sign\Model_AirGesture_Draw\keras_model.h5")
offset = 20
imgSize = 300
index_draw = 0
# serverName = '192.168.165.40'
# serverPort = 2000

serverPort = 80
serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serverSocket.bind(('', serverPort))

# clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# clientSocket.connect((serverName,serverPort))
index = 99
prediction = []
labels = ['BackLeft','BackRight','Backward','Forward','HeadLeft','HeadRight','Left','Right','Stop']
message = 1
# serverSocket.listen(1)
#-------------Drawing Tool Box--------------
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
predicted_class = None
#---------------------------

reg0 = 'S'
reg1 = 'S'
reg2 = 'S'
reg3 = 'S'
message = 'S'
state_application = 0

while True:
    # connectionSocket, addr = serverSocket.accept()
    print('Server is ready to serve...')
    # try:
    while True:
        if state_application == 0:
            success, img = cap.read()
            imgOutput = img.copy()
            hands, img = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgCrop = img[y-offset:y+h + offset, x - offset:x+w+offset]


                imgCropShape = imgCrop.shape

                imgWhite = np.ones((imgSize, imgSize,3),np.uint8)*255
                
                aspectRatio = h/w

                if aspectRatio > 1:
                    k = imgSize/h
                    wCal = math.ceil(k*w)
                    try:
                        imgResize = cv2.resize(imgCrop,(wCal,imgSize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgSize-wCal)/2)
                        imgWhite[:,wGap:wCal+wGap] = imgResize
                        
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    except Exception as e:
                        print(str(e))
                else:
                    k = imgSize/w
                    hCal = math.ceil(k*h)
                    try:
                        imgResize = cv2.resize(imgCrop,(imgSize,hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgSize-hCal)/2)
                        imgWhite[hGap:hCal+hGap,:] = imgResize
                        
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    except Exception as e:
                        print(str(e)) 
                    # print(prediction, index)
            try:  
                cv2.putText(imgOutput, 'Hand Sign Mode', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 5,
                    cv2.LINE_AA)
                cv2.rectangle(imgOutput, (x-offset,y-offset-50), (x+w+offset, y-offset), (0, 255, 0),cv2.FILLED,4)
                cv2.putText(imgOutput, labels[index], (x, y -30), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                cv2.rectangle(imgOutput, (x-offset,y-offset), (x+w+offset, y+h+offset), (0, 255, 0),4)
                cv2.imshow("Image", imgOutput)
            except:
                print("Image Error")
            try: 
                if   (index == 0) & (prediction[0] >= 0.65) : message ='H'
                elif (index == 1) & (prediction[1] >= 0.65): message = 'J'
                elif (index == 2) & (prediction[2] >= 0.65): message = 'B'
                elif (index == 3) & (prediction[3] >= 0.65): message = 'F' 
                elif (index == 4) & (prediction[4] >= 0.65): message = 'G' 
                elif (index == 5) & (prediction[5] >= 0.65): message = 'I' 
                elif (index == 6) & (prediction[6] >= 0.65): message = 'L' 
                elif (index == 7) & (prediction[7] >= 0.65): message = 'R'  
                else: message = 'S'
            except:
                message = 'S'

            reg2 = reg1
            reg1 = reg0
            reg0 = message

            if ((reg0 == reg1) & (reg1 == reg2)):
                if (reg3 != reg2):
                    reg3 = reg2
            # connectionSocket.send(reg3.encode())
            # connectionSocket.recv(1024).decode()
            print('Signal was sent to Car: ', reg3)
            index = 99
            print(prediction)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                state_application = 1
                cv2.destroyAllWindows()
        else:
            cap = cv2.VideoCapture(0)
            points = deque(maxlen=512)
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            is_drawing = False
            is_shown = False
            with mp_hands.Hands(
                    max_num_hands=1,
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
                while cap.isOpened():
                    success, image = cap.read()
                    image = cv2.flip(image, 1)
                    if not success:
                        continue

                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image)

                    # Draw the hand annotations on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y and hand_landmarks.landmark[12].y < \
                                    hand_landmarks.landmark[11].y and hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y:
                                if len(points):
                                    is_drawing = False
                                    is_shown = True
                                    send_mess = 1
                                    canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                                    canvas_gs = cv2.medianBlur(canvas_gs, 9)
                                    canvas_gs = cv2.GaussianBlur(canvas_gs, (5, 5), 0)
                                    # cv2.imshow("Nameless", canvas_gs)
                                    ys, xs = np.nonzero(canvas_gs)
                                    if len(ys) and len(xs):
                                        min_y = np.min(ys)
                                        max_y = np.max(ys)
                                        min_x = np.min(xs)
                                        max_x = np.max(xs)
                                        canvas_gs = cv2.cvtColor(canvas_gs, cv2.COLOR_GRAY2BGR)
                                        cropped_image = canvas_gs[min_y:max_y, min_x: max_x]
                                        # time.sleep(20)
                                        # cropped_image = image[min_y:max_y, min_x: max_x]
                                        cropped_image = cv2.resize(cropped_image, (224, 224), interpolation=cv2.INTER_AREA)
                                        
                                        cropped_image = img_to_array(cropped_image)
                                        cropped_image = np.expand_dims(cropped_image, axis=0)
                                        # cv2.imshow("fff", cropped_image)
                                        # cropped_image = np.asarray(cropped_image, dtype=np.float32).reshape(1, 224, 224, 3)
                                        # Normalize the image array

                                        # Have the model predict what the current image is. Model.predict
                                        # returns an array of percentages. Example:[0.2,0.8] meaning its 20% sure
                                        # it is the first label and 80% sure its the second label.
                                        # cropped_image = cv2.resize(cropped_image, (28, 28))

                                        print("P1")
                                        # cropped_image = np.array(cropped_image, dtype=np.float32)[None, None, :, :]
                                        # print(cropped_image)
                                        #Put your predict model 
                                        prediction_draw = model.predict(cropped_image)
                                        index_draw = np.argmax(prediction_draw)
                                        points = deque(maxlen=512)
                                        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                            else:
                                is_drawing = True
                                is_shown = False
                                points.append((int(hand_landmarks.landmark[8].x*640), int(hand_landmarks.landmark[8].y*480)))
                                for i in range(1, len(points)):
                                    cv2.line(image, points[i - 1], points[i], (0, 255, 0), 2)
                                    cv2.line(canvas, points[i - 1], points[i], (255, 255, 255), 5)
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                            if not is_drawing and is_shown:
                                cv2.putText(image, 'You are drawing {}'.format(CLASSES[index_draw]), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 5,
                                            cv2.LINE_AA)
                                            # "circle","star","triangle","lightning"]
                                if (CLASSES[index_draw] == "circle") & (send_mess == 1): reg3 = "u"
                                elif (CLASSES[index_draw] == "star") & (send_mess == 1): reg3 = "d"
                                elif (CLASSES[index_draw] == "triangle") & (send_mess == 1): reg3 = "s"
                                else: 
                                    reg3 = "S"
                                    # index_draw = 3
                                send_mess = 0
                                # connectionSocket.send(reg3.encode())
                                # connectionSocket.recv(1024).decode()
                            else:
                                cv2.putText(image, 'Drawing Mode', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 5,
                                cv2.LINE_AA)
                    # Flip the image horizontally for a selfie-view display.
                    cv2.imshow('MediaPipe Hands', image)
                    # time.sleep(20)

                    if cv2.waitKey(5) & 0xFF == ord('p'):
                        state_application = 0
                        cv2.destroyAllWindows()
            cap.release()
    # except:
    #     print("Connection Fail")

connectionSocket.close()