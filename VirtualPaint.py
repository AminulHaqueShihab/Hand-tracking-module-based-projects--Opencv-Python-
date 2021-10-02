import cv2
import time
import numpy as np
import HandTrackingModule as htm
import os

folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []
detector = htm.handDetector(detectionCon=0.7)
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
#print(len(overlayList))

brushSize = 10
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

header = overlayList[0]
drawColor = (0,255, 0)
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
        # Check Which finders are up
        fingers = detector.fingersUp()
            #print(fingers)

        #If selection mode, 2 fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            #checking for clicks
            if y1 < 125:
                if 100< x1 < 250:
                    header = overlayList[0]
                    drawColor = (0,255, 0)
                elif 300 < x1 < 450:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 500 < x1 < 650:
                    header = overlayList[2]
                    drawColor = (0, 0, 255)
                elif 700 < x1 < 850:
                    header = overlayList[3]
                    drawColor = (0, 255, 255)
                elif 900 < x1 < 1050:
                    header = overlayList[4]
                    drawColor = (255, 255, 0)
                elif 1100 < x1 < 1250:
                    header = overlayList[5]
                    drawColor = (255, 255, 255)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), (drawColor), cv2.FILLED)


        #If drawing mode, Index finger is up
        if fingers[1] and fingers[2] == False:

            cv2.circle(img, (x1, y1), 10, (drawColor), cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (255,255,255):
                cv2.line(img, (xp, yp), (x1, y1), (0,0,0), 150)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), (0,0,0), 150)
            else:
                cv2.line(img, (xp, yp),(x1, y1), drawColor, brushSize)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushSize)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    #setting the header image
    img[0:125, 0:1280] = header
    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)