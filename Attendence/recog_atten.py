import cv2 as cv
import numpy as np
import os

cap = cv.VideoCapture(0)

while True:
    isTrue, Frame = cap.read()

    cv.imshow('Student', Frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()

cv.destroyAllWindows()