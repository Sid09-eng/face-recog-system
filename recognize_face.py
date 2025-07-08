import cv2 # type: ignore
import numpy as np # type: ignore
from joblib import load # type: ignore

MODEL_DIR = 'models'
pca = load(f'{MODEL_DIR}/pca_model.joblib')
knn = load(f'{MODEL_DIR}/knn_classifier.joblib')
le = load(f'{MODEL_DIR}/label_encoder.joblib')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100)).flatten().reshape(1, -1)
        face_pca = pca.transform(face_img)
        pred = knn.predict(face_pca)
        name = le.inverse_transform(pred)[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
