'''
Video e-KYC by facial feature tracking
Author: Azfar Lari

This is the complete code for the full process
with the implementation of the random action detection from the user

Will require some refactoring and rectification of syntax errors
A lot of errors may occur, A LOT
'''

import cv2
import numpy as np
from mtcnn import MTCNN
from imutils.video import VideoStream
from eye_status import *
import re
import random

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

    print("[LOG] Opening webcam ...")
    video_capture = VideoStream(src=0).start()
    detector = MTCNN()
    model = load_model()

    return (model, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, video_capture, detector)


def isBlinking(history, maxTimes):
    regex = re.compile("10+1")
    if len(re.findall(regex, history)) >= maxTimes:
        return True
    # if history.count("0")>=maxFrames:
    # return True
    return False

def display(frame):
    cv2.imshow("frame", frame)
    cv2.waitKey(1)

def going_left(video_capture, face_detector, left_eye_detector, right_eye_detector, detector):
    print("Please turn left")
    eye_pos_count = 1
    eye_pos = 0
    count=0
    while True:
        frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print("running haar cascade")
        # Detect faces
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # print("Faces detected=", len(faces))
        if not len(faces) == 1:
            print("Only one face allowed in front of the camera")
            continue
        else:
            # for each detected face
            for (x, y, w, h) in faces:
                yes = False
                face = frame[y:y + h, x:x + w]
                gray_face = gray[y:y + h, x:x + w]

                left_face = frame[y:y + h, x + int(w / 2):x + w]
                left_face_gray = gray[y:y + h, x + int(w / 2):x + w]

                right_face = frame[y:y + h, x:x + int(w / 2)]
                right_face_gray = gray[y:y + h, x:x + int(w / 2)]

                # Eyes detection
                # try detecting eyes using left and right_eye_detector
                # which can detect open and closed eyes

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

            ex, ey, ew, eh = left_eye[0]
            ex1, ey1, ew1, eh1 = right_eye[0]
            #Put in try block for any exception due to non-detection of eyes
            try:
                if eye_pos_count <= 1:
                    eye_pos = ey
                    print("please turn your face left")
                eye_pos_count += 1
                #else:
                #   eye_pos-=eye
                if eye_pos - ey > 0 and eye_pos - ey < 2:
                    count+=1
                    print("Going left")
                else:
                    eye_pos = ey
                display(frame)
            except:
                display(frame)
                continue
            ## Adjust this value in accordance with the distance travelled by the eyes in the direction
            if count>3:
                return True
                break
    # Add a time limit for the user to make this action
    return False

def going_right(video_capture, face_detector, left_eye_detector, right_eye_detector, detector):
    print("Please turn right")
    eye_pos_count = 1
    eye_pos = 0
    count = 0
    while True:
        frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print("running haar cascade")
        # Detect faces
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # print("Faces detected=", len(faces))
        if not len(faces) == 1:
            print("Only one face allowed in front of the camera")
            continue
        else:
            # for each detected face
            for (x, y, w, h) in faces:
                yes = False
                face = frame[y:y + h, x:x + w]
                gray_face = gray[y:y + h, x:x + w]

                left_face = frame[y:y + h, x + int(w / 2):x + w]
                left_face_gray = gray[y:y + h, x + int(w / 2):x + w]

                right_face = frame[y:y + h, x:x + int(w / 2)]
                right_face_gray = gray[y:y + h, x:x + int(w / 2)]

                # Eyes detection
                # try detecting eyes using left and right_eye_detector
                # which can detect open and closed eyes

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
                ex, ey, ew, eh = left_eye[0]
                ex1, ey1, ew1, eh1 = right_eye[0]
            # Put in try block for any exception due to non-detection of eyes
            try:
                if eye_pos_count <= 1:
                    eye_pos = ey
                    print("please turn your face left")
                eye_pos_count += 1
                #else:
                #   eye_pos-=eye
                if eye_pos - ey < 0 and eye_pos - ey > -2:
                    count += 1
                    print("Going right")
                else:
                    eye_pos = ey
                display(frame)
            except:
                display(frame)
                continue
            ## Adjust this value in accordance with the distance travelled by the eyes in the direction
            if count > 3:
                return True
                break
    # Add a time limit for the user to make this action


def going_up(video_capture, face_detector, left_eye_detector, right_eye_detector, detector):
    print("Please turn up")
    eye_pos_count = 1
    eye_pos = 0
    count = 0
    while True:
        frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print("running haar cascade")
        # Detect faces
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # print("Faces detected=", len(faces))
        if not len(faces) == 1:
            print("Only one face allowed in front of the camera")
            continue
        else:
            # for each detected face
            for (x, y, w, h) in faces:
                yes = False
                face = frame[y:y + h, x:x + w]
                gray_face = gray[y:y + h, x:x + w]

                left_face = frame[y:y + h, x + int(w / 2):x + w]
                left_face_gray = gray[y:y + h, x + int(w / 2):x + w]

                right_face = frame[y:y + h, x:x + int(w / 2)]
                right_face_gray = gray[y:y + h, x:x + int(w / 2)]

                # Eyes detection
                # try detecting eyes using left and right_eye_detector
                # which can detect open and closed eyes

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
            ex, ey, ew, eh = left_eye[0]
            ex1, ey1, ew1, eh1 = right_eye[0]
            # Put in try block for any exception due to non-detection of eyes
            try:
                if eye_pos_count <= 1:
                    eye_pos = ex
                    
                eye_pos_count += 1
                #else:
                #   eye_pos-=eye
                if eye_pos - ex > 0 and eye_pos - ex < 2:
                    count += 1
                    print("Going up")
                else:
                    eye_pos = ex
                display(frame)
            except:
                display(frame)
                continue
            ## Adjust this value in accordance with the distance travelled by the eyes in the direction
            if count > 3:
                return True
                break
        # Add a time limit for the user to make this action


def going_down(video_capture, face_detector, left_eye_detector, right_eye_detector, detector):
    print("Please Turn down")
    eye_pos_count = 1
    eye_pos = 0
    count = 0
    while True:
        frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print("running haar cascade")
        # Detect faces
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # print("Faces detected=", len(faces))
        if not len(faces) == 1:
            print("Only one face allowed in front of the camera")
            continue
        else:
            # for each detected face
            for (x, y, w, h) in faces:
                yes = False
                face = frame[y:y + h, x:x + w]
                gray_face = gray[y:y + h, x:x + w]

                left_face = frame[y:y + h, x + int(w / 2):x + w]
                left_face_gray = gray[y:y + h, x + int(w / 2):x + w]

                right_face = frame[y:y + h, x:x + int(w / 2)]
                right_face_gray = gray[y:y + h, x:x + int(w / 2)]

                # Eyes detection
                # try detecting eyes using left and right_eye_detector
                # which can detect open and closed eyes

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
            # Put in try block for any exception due to non-detection of eyes
            try:
                if eye_pos_count <= 1:
                    eye_pos = ex
                    
                eye_pos_count += 1
                #else:
                #   eye_pos-=eye
                if eye_pos - ex < 0 and eye_pos - ex > -2:
                    count += 1
                    print("Going down")
                else:
                    eye_pos = ex
                display(frame)
            except:
                display(frame)
                continue
            ## Adjust this value in accordance with the distance travelled by the eyes in the direction
            if count > 3:
                return True
    # Add a time limit for the user to make this action



def blink_test(model, video_capture, face_detector, left_eye_detector, right_eye_detector, detector):
    print("Please Start Blinking")
    while True:
        frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print("running haar cascade")
        # Detect faces
        faces = detector.detect_faces(rgb)
        x, y, w, h = faces[0]["box"]

        if not len(faces) == 1:
            # print("Only one face allowed in front of the camera")
            continue
        else:
            # for each detected face
            try:
                yes = False
                face = frame[y:y + h, x:x + w]
                gray_face = gray[y:y + h, x:x + w]
            except:
                print("error after face")
                break

            global eyes_detected

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
            eye_status = "1"
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
            if eye_status == "0":
                print("Blink")
            # print(eyes_detected)
            # Each time, we check if the person has blinked
            # the 2nd parameter is the total number of blinks permitted
            display(frame)
            if isBlinking(eyes_detected, 2):
                return (True)
            ##Add a time limit as needed
    return (False)

def start_actions(model, video_capture, face_detector, left_eye_detector, right_eye_detector, detector):
    actions=random.sample(["up", "down", "left", "right", "blink"], 2)
    result = []
    for i in actions:
        if i == "blink":
            x=blink_test(model, video_capture, face_detector, left_eye_detector, right_eye_detector, detector)
            result.append(x)
        elif i =="left":
            x=going_left(video_capture, face_detector, left_eye_detector, right_eye_detector, detector)
            result.append(x)
        elif i == "right":
            x=going_right(video_capture, face_detector, left_eye_detector, right_eye_detector, detector)
            result.append(x)
        elif i == "up":
            x=going_up(video_capture, face_detector, left_eye_detector, right_eye_detector, detector)
            result.append(x)
        else:
            x=going_down(video_capture, face_detector, left_eye_detector, right_eye_detector, detector)
            result.append(x)

    if False in result:
        return(False)
    else:
        return(True)


def detect_features(model, video_capture, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, detector):
    while True:
        frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print("running haar cascade")
        # Detect faces
        faces = detector.detect_faces(rgb)
        x, y, w, h = faces[0]["box"]
        # print("Faces detected=", len(faces))
        if not len(faces) == 1:
            print("Only one face allowed in front of the camera")
            continue
        else:
            # for each detected face
            try:
                yes = False
                face = frame[y:y + h, x:x + w]
                gray_face = gray[y:y + h, x:x + w]

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
                if not len(open_eyes_glasses) == 0:
                    print("Please remove glasses")
                    display(frame)

                # try detecting eyes using left and right_eye_detector
                # which can detect open and closed eyes
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
            except:
                print("Error After Face")

        result = start_actions(model, video_capture, face_detector, left_eye_detector, right_eye_detector, detector)
        if result:
            print("Real Person")
        else:
            print("Not a real person")
        video_capture.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    (model, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, video_capture, detector) = init()
    eye_pos_count = 1
    eye_pos = 0
    print("[STARTING TRACKING]")
    detect_features(model, video_capture, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, detector)
