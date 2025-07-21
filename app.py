import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('/Users/kavydave/Desktop/ML/OpenCV/facedetaction/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/kavydave/Desktop/ML/OpenCV/facedetaction/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('/Users/kavydave/Desktop/ML/OpenCV/facedetaction/haarcascade_smile.xml')
cap = cv2.VideoCapture(0)

while True:
    ret , frame = cap.read()

    if not ret:
        print("Failed")
        break

    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray  , 1.1 , 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame , (x,y) , (x+w , y+h) , (255,0,0) , 2)
        cv2.putText(frame , "Face" , (x,y-10) , cv2.FONT_HERSHEY_SIMPLEX , 0.9 , (255,0,0) , 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray , 1.7 , 22)
        if len(eyes) >0:
            cv2.putText(roi_color , "Eye" , (x,y+h+20) , cv2.FONT_HERSHEY_SIMPLEX , 0.9 , (0,255,0) , 2)
        
        smile = smile_cascade.detectMultiScale(roi_gray , 1.8 , 22)
        if len(smile) > 0:
            cv2.putText(roi_color , "Smile" , (x,y+h+10) , cv2.FONT_HERSHEY_SIMPLEX , 0.9 , (0,255,255) , 2)
    cv2.imshow("Face Detection" , frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()