import cv2 as cv
import numpy as np
import os
import csv

attendence = []
info = {}

cap = cv.VideoCapture(0)

students = ['ALL STUDENTS NAME']

recognizer = cv.face.LBPHRecognizer_create()

recognizer.read('Trained.yml')
haar = cv.CascadeClassifier('haar.xml')

while True:
    isTrue, Frame = cap.read()
    face_rect = haar.detectMultiScale(Frame, 1.1, 4)
    gray = cv.cvtColor(Frame, cv.COLOR_BGR2GRAY)
    for (x,y, h,w) in face_rect:
        face_ir = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_ir)
    info = {'Student':students[label],'Attendence':'Present'}
    attendence.append(info)
        # cv.rectangle(Frame,(x,y),(x+w,y+h),(0,2550,0),1)
    # cv.imshow('Student', Frame)


    if cv.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()

cv.destroyAllWindows()

column_names = ['Students', 'Attendence']

with open('attendence.csv') as attendence_sheet:
    writer = csv.DictWriter(attendence_sheet, fieldnames=column_names)
    writer.writeheader()
    writer.writerows(info)