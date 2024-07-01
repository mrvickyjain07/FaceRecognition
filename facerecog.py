import cv2
import os
import numpy as np

haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
datasets = 'dataset'

print('Training...')
(images, labels, name, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        name[id] = subdir
        subject_path = os.path.join(datasets, subdir)
        for filename in os.listdir(subject_path):
            path = os.path.join(subject_path, filename)
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

(images, labels) = [np.array(lis) for lis in [images, labels]]
print(images, labels)

(width, height) = (130, 100)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

webcam = cv2.VideoCapture(0)
cnt = 0

while True:
    _, img = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        prediction = model.predict(face_resize)
        
        if prediction[1] < 500:
            cnt = 0
            person_name = name[prediction[0]]
            confidence = int(100 * (1 - (prediction[1] / 300)))
            cv2.putText(img, f'{person_name} {confidence}%', (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cnt += 1
            cv2.putText(img, 'Unknown', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("Unknown.jpg", img)
                cnt = 0

    cv2.imshow('FaceRecognition', img)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
