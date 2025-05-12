!pip install easyocr opencv-python-headless
import cv2
import easyocr
import numpy as np

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Load Haar cascade for plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Open video file (or 0 for webcam)
video_path = 'traffic.mp4'  # Replace with your own video file
cap = cv2.VideoCapture(video_path)

# Store unique plate numbers
detected_plates = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plate_img = gray[y:y + h, x:x + w]

        # Apply OCR on plate region
        result = reader.readtext(plate_img)
        for detection in result:
            text = detection[1]
            text = text.strip()

            # Filter possible plate formats (simple rule)
            if 5 <= len(text) <= 15 and text not in detected_plates:
                detected_plates.add(text)
                print(f"Detected Plate: {text}")

            # Draw text on frame
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 0, 255), 2)

    # Show frame
    cv2.imshow("Number Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Show all unique detected plates
print("\n? All Detected Plates:")
for plate in detected_plates:
    print(plate)
