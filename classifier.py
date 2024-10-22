import cv2
import numpy as np
import os

#######################################################
def distance(v1, v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
    dist=[]
    for i in range(train.shape[0]):
        ix = train[i,:-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d,iy])

    dk = sorted(dist, key = lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]
#######################################################


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

skip = 0
face_data = []
labels = []
dataset_path = "./data/"

class_id = 0 # Labels for the given file
names = {} #Mapping between id and name

# Data Loading and Data preparation
for file in os.listdir(dataset_path):
    if file.endswith('.npy'):
        names[class_id] = file[:-4]
        print("Loaded " + file)
        data_item = np.load(dataset_path+file)
        face_data.append(data_item)

        # Create labels for the class
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1,1))
training_set = np.concatenate((face_dataset, face_labels), axis=1)

# Loading the video

while True:
    ret, frame = cap.read()
    if ret == False: 
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, 1.3 ,5)
    for face in faces:
        x,y,w,h = face
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100,100))
        
        # Predicting the face
        
        out = knn(training_set, face_section.flatten())
        prediction = names[int(out)]
        cv2.putText(gray_frame, prediction, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
        cv2.rectangle(gray_frame, (x,y), (x+w, y+h), (0,0,0), 2)
    cv2.imshow("Video Classifier", gray_frame)

    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
