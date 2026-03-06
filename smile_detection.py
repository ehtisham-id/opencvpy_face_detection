import cv2 as cv
import numpy as np
import os

smile_cascade = cv.CascadeClassifier('smile_cascade.xml')

if smile_cascade.empty():
    print("Error loading cascade file. Check the path.")
    exit()

# Start video capture
vid = cv.VideoCapture(0)

while True:
    ret, frame = vid.read()

    if not ret:
        print('Frame reading error')
        break
    
    # Corrected function name for grayscale conversion
    grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect smiles (note that this should detect smiles, not faces)
    smile = smile_cascade.detectMultiScale(
        grey_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around detected smiles
    for (x, y, w, h) in smile:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Display the frame
    cv.imshow('Smile Detector', frame)
    
    # Exit on pressing 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
vid.release()
cv.destroyAllWindows()
