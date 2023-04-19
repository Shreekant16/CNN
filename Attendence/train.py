import cv2 as cv
import os
import numpy as np

dataset = r''

names = []
for name in os.listdir(dataset):
    names.append(name)

haar = cv.CascadeClassifier('haar.xml')

# print(name)
labels = []
features = []
for i in names:
    file_path = os.path.join(dataset,i)
    label = names.index(i)
    for j in os.listdir(file_path):
        img_path = os.path.join(file_path, j)
        img = cv.imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        face_rect = haar.detectMultiScale(gray,1.1,4)
        for (x,y,h,w) in face_rect:
            face_ir = gray[y:y+h, x:x+w]
            features.append(face_ir)
            labels.append(label)

features = np.array(features, dtype='object')
labels = np.array(labels)

np.save('Features.npy',features)
np.save('labels.npy', labels)

recognizer = cv.face.LBPHFaceRecognizer_create()

recognizer.train(features, labels)

recognizer.save('Trained.yml')