'''
Video e-KYC by facial feature tracking
Author: Azfar Lari
'''

import os
import cv2
import numpy as np
from imutils.video import VideoStream
from eye_status import *
import re
from mtcnn import MTCNN

eyes_detected = ""


def init():
    face_cascPath = 'haarcascade_frontalface_alt.xml'
    # face_cascPath = 'lbpcascade_frontalface.xml'

    open_eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'
    left_eye_cascPath = 'haarcascade_lefteye_2splits.xml'
    right_eye_cascPath = 'haarcascade_righteye_2splits.xml'
    dataset = 'faces'

    face_detector = cv2.CascadeClassifier(face_cascPath)
    open_eyes_detector = cv2.CascadeClassifier(open_eye_cascPath)
    left_eye_detector = cv2.CascadeClassifier(left_eye_cascPath)
    right_eye_detector = cv2.CascadeClassifier(right_eye_cascPath)

    detector = MTCNN()
    print("[LOG] Opening webcam ...")
    video_capture = VideoStream(src=0).start()

    model = load_model()

    return (model, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, video_capture, detector)


def isBlinking(history, maxTimes):
    regex = re.compile("10+1")
    if len(re.findall(regex, history)) >= maxTimes:
        return True
    # if history.count("0")>=maxFrames:
    # return True
    return False


def detect_and_display(model, video_capture, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, detector):
    eye_pos_count = 1
    text = ""
    eye_pos = 0
    while True:
        frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #print("running haar cascade")
        # Detect faces
        faces = detector.detect_faces(rgb)
        #print(faces[0]["box"])
        x, y, w, h = faces[0]["box"]
        #print("Faces detected=", len(faces))
        if not len(faces)==1:
            #print("Only one face allowed in front of the camera")
            text = "Only one face allowed in front of the camera"
            continue
        else:
            # for each detected face
            
            try:
                yes=False
                face = frame[y:y + h, x:x + w]
                gray_face = gray[y:y + h, x:x + w]

                eyes = []
                global eyes_detected
                # Eyes detection
                # check first if eyes are open (with glasses taking into account)
                open_eyes_glasses = open_eyes_detector.detectMultiScale(
                    gray_face,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                # if open_eyes_glasses detect eyes then they are open
                if len(open_eyes_glasses) == 2:
                    eyes_detected += '1'
                    for (ex, ey, ew, eh) in open_eyes_glasses:
                        cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                # otherwise try detecting eyes using left and right_eye_detector
                # which can detect open and closed eyes
                else:
                    # separate the face into left and right sides
                    left_face = frame[y:y + h, x + int(w / 2):x + w]
                    left_face_gray = gray[y:y + h, x + int(w / 2):x + w]

                    right_face = frame[y:y + h, x:x + int(w / 2)]
                    right_face_gray = gray[y:y + h, x:x + int(w / 2)]

                    # Detect the left eye
                    left_eye = left_eye_detector.detectMultiScale(
                        left_face_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    # Detect the right eye
                    right_eye = right_eye_detector.detectMultiScale(
                        right_face_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    eye_status = '1'  # we suppose the eyes are open

                    # For each eye check wether the eye is closed.
                    # If one is closed we conclude the eyes are closed
                    for (ex, ey, ew, eh) in right_eye:
                        color = (0, 255, 0)
                        pred = predict(right_face[ey:ey + eh, ex:ex + ew], model)
                        if pred == 'closed':
                            eye_status = '0'
                            color = (0, 0, 255)
                        cv2.rectangle(right_face, (ex, ey), (ex + ew, ey + eh), color, 2)
                    for (ex, ey, ew, eh) in left_eye:
                        color = (0, 255, 0)
                        pred = predict(left_face[ey:ey + eh, ex:ex + ew], model)
                        if pred == 'closed':
                            eye_status = '0'
                            color = (0, 0, 255)
                        cv2.rectangle(left_face, (ex, ey), (ex + ew, ey + eh), color, 2)
                    eyes_detected += eye_status
                #print(eyes_detected)
                # Each time, we check if the person has blinked
                # the 2nd parameter is the total number of blinks permitted
                if isBlinking(eyes_detected, 4):
                    yes = True
                    print('Real Person')
                    break
            except:
                print("Error after face")
            try:
                ey, ex = faces[0]["keypoints"]["nose"]
                #print("features=", faces[0]["keypoints"])
                               

                if eye_pos_count <= 1:
                    eye_pos = ey
                eye_pos_count += 1
                # else:
                #	eye_pos-=eye
                if eye_pos - ey > 3:
                    eye_pos = ey
                    text = "Going left"
                    #print("Going right")
                elif eye_pos - ey < -3:
                    eye_pos = ey
                    text = "Going right"
                    #print("Going left")
                else:
                    text = "Face is steady"
                    #print("Face is steady")
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Face", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if yes:
                    video_capture.stop()
                    cv2.destroyAllWindows()
                    break
            except:
                continue

if __name__ == "__main__":
    (model, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, video_capture, detector) = init()
    eye_pos_count = 1
    eye_pos = 0
    print("[STARTING TRACKING]")
    detect_and_display(model, video_capture, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, detector)
