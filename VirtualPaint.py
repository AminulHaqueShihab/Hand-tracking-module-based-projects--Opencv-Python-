import cv2
import time
import numpy as np
import HandTrackingModule as htm
import os

folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []
detector = htm.handDetector(detectionCon=0.85)
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
#print(len(overlayList))

header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    #find landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        #tip of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #cx, cy = (x1 + x2) // 2, (y1 + y2) // 2


    #setting the header image
    img[0:125, 0:1280] = header
    cv2.imshow("Image", img)
    cv2.waitKey(1)