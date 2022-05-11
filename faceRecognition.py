import numpy as np

import cv2

from sklearn.neighbors import KNeighborsClassifier

data = np.load("face_data.npy")


x = data[:, 1:].astype(int)
y = data[:, 0]

model = KNeighborsClassifier()
model.fit(x, y)

cap = cv2.VideoCapture(1)

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while True:

    ret, frame = cap.read()

    if ret:
        faces = detector.detectMultiScale(frame)

        for face in faces:
            x, y, w, h = face
            cut = frame[y:y + h, x:x + w]
            fix = cv2.resize(cut, (100, 100))
            gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)
            out = model.predict([gray.flatten()])

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
            cv2.putText(frame,str(out[0]),(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        cv2.imshow("My Screen", frame)


    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()

