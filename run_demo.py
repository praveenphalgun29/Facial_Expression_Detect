import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model_file_improved.keras')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))

        result = model.predict(reshaped, verbose=0)
        label_index = np.argmax(result, axis=1)[0]
        
        predicted_expression = labels_dict[label_index]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, predicted_expression, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Facial Expression Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
