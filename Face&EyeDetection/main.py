import cv2 as cv
import numpy as np
haar_cascade_face = cv.CascadeClassifier('haar_face.xml')
haat_cascade_eye = cv.CascadeClassifier('haar_eye.xml')
cap = cv.VideoCapture(0)
while True:
    isTrue, frame = cap.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    face_det = haar_cascade_face.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in face_det:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        face_ir = gray[y:y+h,x:x+w]
        face_ir_c = frame[y:y+h,x:x+w]
        face_eye = haat_cascade_eye.detectMultiScale(face_ir,1.1,4)
        for (x,y,w,h) in face_eye:
            cv.rectangle(face_ir_c,(x,y),(x+w,y+h),(0,0,255),2)

    cv.imshow('Video',frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break
cap.release()
cv.destroyAllWindows()