import cv2
import os
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
import pandas as pd

# Load known faces
known_faces = {}
for file in os.listdir("known_faces"):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        name = os.path.splitext(file)[0]
        path = os.path.join("known_faces", file)
        img = cv2.imread(path)
        if img is not None:
            known_faces[name] = cv2.resize(img, (100, 100))

# Attendance data
attendance = []
marked_names = set()

# Function to compare two images
def is_match(img1, img2, threshold=0.7):
    img1 = cv2.resize(img1, (100, 100))
    img2 = cv2.resize(img2, (100, 100))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score >= threshold, score

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'n' to scan face, 'q' to quit and save attendance.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show live feed
    cv2.imshow("Live Camera - Press 'n' to mark, 'q' to quit", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('n'):
        faces, confidences = cv.detect_face(frame)

        for face in faces:
            x1, y1 = face[0], face[1]
            x2, y2 = face[2], face[3]
            face_crop = frame[y1:y2, x1:x2]

            best_score = 0
            best_name = "Unknown"

            for name, known_img in known_faces.items():
                match, score = is_match(known_img, face_crop)
                if score > best_score:
                    best_score = score
                    best_name = name

            print(f"Detected: {best_name} | Score: {best_score:.2f}")

            if best_score >= 0.14 and best_name not in marked_names:
                now = datetime.now()
                attendance.append([best_name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
                marked_names.add(best_name)
                print(f"✅ Attendance marked for {best_name}")
            elif best_name in marked_names:
                print(f"⚠️ {best_name} already marked.")
            else:
                print("❌ Match not strong enough.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save attendance
if attendance:
    df = pd.DataFrame(attendance, columns=["Name", "Date", "Time"])
    df.to_csv("attendance.csv", index=False)
    print("✅ Attendance saved to attendance.csv")
else:
    print("⚠️ No attendance recorded.")
