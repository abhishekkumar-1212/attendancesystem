import cv2
import numpy as np
import csv
import face_recognition
from datetime import datetime
import pyttsx3

# Initialize the TTS engine
tts_engine = pyttsx3.init()

video_captured = cv2.VideoCapture(0)

# Load known faces
harry_image = face_recognition.load_image_file("faces/harry.jpg")
harry_encoding = face_recognition.face_encodings(harry_image)[0]

rohan_image = face_recognition.load_image_file("faces/rohan.jpg")
rohan_encoding = face_recognition.face_encodings(rohan_image)[0]

preeti_image = face_recognition.load_image_file("faces/preeti.jpg")
preeti_encoding = face_recognition.face_encodings(preeti_image)[0]

vanya_image = face_recognition.load_image_file("faces/vanya.jpg")
vanya_encoding = face_recognition.face_encodings(vanya_image)[0]

arav_image = face_recognition.load_image_file("faces/arav.jpg")
arav_encoding = face_recognition.face_encodings(arav_image)[0]

shanu_image = face_recognition.load_image_file("faces/shanu.jpg")
shanu_encoding = face_recognition.face_encodings(shanu_image)[0]

known_face_encodings = [harry_encoding, rohan_encoding, preeti_encoding, vanya_encoding, arav_encoding, shanu_encoding]
known_face_names = ["Abhishek", "Abhishek gola", "Preeti DIDI", "vanya", "Arav bhai", "Shanu priya Di"]

# List of expected students
students = known_face_names.copy()

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Create a CSV file
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_captured.read()  # Capture video frame
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize the faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Add text if a person is present
            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name + " Present ", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_date, current_time])
                    
                    # Speak the name of the present student
                    tts_engine.say(name)
                    tts_engine.runAndWait()

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(10) & 0xFF == ord("e"):
        break

# Release video capture and close all windows
video_captured.release()
cv2.destroyAllWindows()
f.close()
