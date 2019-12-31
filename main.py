# Imports
import sys
import cv2

# Create face cascade
cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

# Set video capture source
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = video_capture.read()

    # Convert to grayscale for image processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Search for faces according to the cascade
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30)
    )

    # Draw a rectangle around all faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # Wait for q button to be pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
video_capture.release()
cv2.destroyAllWindows()
