#!/usr/bin/env python

import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    
    for (x,y,w,h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        eye = eye_cascade.detectMultiScale(roi_color)
        
        #for (sx,sy,sw,sh) in eye:
            #cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0,255,0),4)
            #cv2.circle(roi_color,(sx+sw/2,sy+sh/2), 10, (0,0,255),3)
        kernel = np.ones((40,40), np.float32)/1600
        
        blurr = cv2.filter2D(roi_color,-1, kernel)
        #cv2.imshow('Face Finder Blurred',blurr)

        frame[y:y+h, x:x+w] = blurr
    cv2.imshow('Face Finder ',frame)

    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #lower_red = np.array([0,0,0])
    #upper_red = np.array([255,255,255])

    #mask = cv2.inRange(hsv, lower_red, upper_red)
    #res = cv2.bitwise_and(frame, frame, mask = mask)

      
    
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
