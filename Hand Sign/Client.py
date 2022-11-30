import socket

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
classifier = Classifier("D:\CODE\MMT\A1_simple_web_server\Raspberry_Connection\Model\keras_model.h5",
                        "D:\CODE\MMT\A1_simple_web_server\Raspberry_Connection\Model\labels.txt")
offset = 20
imgSize = 300

serverName = '127.0.0.1'
serverPort = 80



clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientSocket.connect((serverName,serverPort))
index = 5
message = 1
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y-offset:y+h + offset, x - offset:x+w+offset]
        # if (x - offset > 0) & (y - offset > 0):
        #     cv2.imshow("ImageCrop", imgCrop)

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
                prediction, index = classifier.getPrediction(imgWhite)
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
                prediction, index = classifier.getPrediction(imgWhite)
            except Exception as e:
                print(str(e))
            # print(prediction, index)
        
        cv2.imshow("ImageWhite", imgWhite)  

    cv2.imshow("Image", img)

    if index == 0: message = 'left'
    elif index == 1: message = 'right'
    elif index == 2: message = 'forward'
    elif index == 3: message = 'backward'
    else: message = 'stop'
    

    clientSocket.send(message.encode())
    clientSocket.recv(1024)
    print('Signal was sent to Raspberry: ', message)
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    time.sleep(0.25)
clientSocket.close()