import cv2 as cv
import numpy as np
import os

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

haar_cascade = cv.CascadeClassifier('haar_cascade.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('trained.yml')
img  = cv.imread(r'C:\Users\SUHAS\OneDrive\Desktop\Faces\train\Elton John/5.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_rect = haar_cascade.detectMultiScale(gray, 1.1,4)
for (x,y,h,w) in face_rect:
    face_ir = gray[y:y+h,x:x+h]
    label, confidence = recognizer.predict(face_ir)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv.putText(img,f'{people[label]}',(20,20),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

cv.imshow('Recognised',img)
print(f'Confidence is {confidence}')
print(f'Person is {people[label]}')
cv.waitKey(0)