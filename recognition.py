import cv2 as cv 
import numpy as np
import face_recognition as fr

known_image = fr.load_image_file('known.jpg')
face_landmark = fr.face_landmarks(known_image)

known_encoding = fr.face_encodings(known_image)[0]

vid = cv.VideoCapture(0)

while True:
    ret, frame = vid.read()
    
    if not ret:
        print("Frame reading error")
        break

    unknown_encoding = fr.face_encodings(frame)[0]
    
    result = fr.compare_faces([known_encoding], unknown_encoding )
    
    print(result)
    # print(face_landmark)
    
    
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv.destroyAllWindows()