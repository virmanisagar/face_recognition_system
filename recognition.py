import cv2
import numpy as np

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

#id_2_name = {1:'sagar', 2:'Roshan', }

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

"""
1 - Starts capturing frames from the camera object
2 - Convert it to Gray Scale
3 - Detect and extract faces from the images
4 - Use the recognizer to recognize the Id of the user
5 - Put predicted Id/Name and Rectangle on detected face
"""

while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),5)
        Id = recognizer.predict(gray[y:y+h,x:x+w])
        conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf<50):
            if(Id==1):
                Id="Sagar"
            elif(Id==2):
                Id="Something"
            elif(Id==3):
                Id="Unkn"
            elif(Id==9):
                Id="Aryan"
            elif(Id==11):
                Id="Vishal"
            elif(Id==12):
                Id="Abhijeet"
            else:
                Id="Unknown"       
    	cv2.putText(im, str(Id), (x,y+h), font, 4, 255)
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
