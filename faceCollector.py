import cv2

import numpy as np

import os

cap = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

name = input("Enter your name: ")

frames = []
outputs = []


while True:

    ret, frame = cap.read()

    if ret:
        faces = detector.detectMultiScale(frame)
        print(faces)

        for face in faces:
            x, y, w, h = face
            cut = frame[y:y + h, x:x + w]

            fix = cv2.resize(cut, (100, 100))
            gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)

            # frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
        cv2.imshow("My Screen", frame)



    key = cv2.waitKey(1)

    if key == ord("q"):
        break

    if key == ord("c"):
        # cv2.imwrite(name+".jpg",frame)

        frames.append(gray.flatten())
        outputs.append([name])

x = np.array(frames)
y = np.array(outputs)

data = np.hstack([y, x])

f_name = "face_data.npy"

if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old, data])

np.save(f_name, data)

cap.release()
cv2.destroyAllWindows()
