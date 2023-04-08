import cv2 as cv
import numpy as np
import os

data = r'C:\Users\SUHAS\OneDrive\Desktop\Faces\train'
people = []
for i in os.listdir(data):
    people.append(i)
haar_cascade = cv.CascadeClassifier('haar_cascade.xml')
# print(people)
features = []
labels = []
def train():
    for i in people:
        folder = os.path.join(data,i)
        label = people.index(i)
        for img_paths in os.listdir(folder):
            img_path = os.path.join(folder,img_paths)
            img = cv.imread(img_path)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            face_rect = haar_cascade.detectMultiScale(gray,1.1,4)
            for (x,y,h,w) in face_rect:
                face_ir = gray[y:y+h,x:x+w]
                features.append(face_ir)
                labels.append(label)

train()
# print(len(features))
# print(len(labels))
features = np.array(features)
labels = np.array(labels)
recognizer = cv.face.LBPHFaceRecognizer_create()
print('Training--------------------')
recognizer.train(features,labels)
recognizer.save('trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)

