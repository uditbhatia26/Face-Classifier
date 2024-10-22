import cv2
import numpy as np
import os

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load face cascade for detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

skip = 0
face_data = []
path = './data/'
username = input("Enter your name: ")

# Ensure the directory exists
if not os.path.exists(path):
    os.makedirs(path)

# Capture images from webcam
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, 1.05, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw bounding boxes and process the largest face
    for face in faces:
        x, y, w, h = face
        cv2.rectangle(gray_frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

        offset = 10
        face_section = gray_frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        # Save every 10th face
        skip += 1
        if skip % 10 == 0:
            face_data.append(face_section)
            print(f"Captured face {len(face_data)}")

        # Display the face and video frame
        cv2.imshow("Face Section", face_section)

    cv2.imshow("Video", gray_frame)

    # Break on 'q' key press
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Convert list to numpy array and save
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))

np.save(path + username + ".npy", face_data)
print(f"Data saved successfully as {path + username + '.npy'}")

# Release resources
cap.release()
cv2.destroyAllWindows()
