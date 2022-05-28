import face_recognition
import cv2
import os
import glob
import numpy as np

def chargement_images(chemin):

    known_face_encodings = []
    known_face_names = []

    images_path = glob.glob(os.path.join(chemin, "*.*"))
    print("{} images trouvés à encoder.".format(len(images_path)))

    for img_path in images_path:
        img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        basename = os.path.basename(img_path)
        (filename, ext) = os.path.splitext(basename)

        img_encoding = face_recognition.face_encodings(rgb_img)[0]

        known_face_encodings.append(img_encoding)
        known_face_names.append(filename)
    print("Encodage fini.")
    return known_face_encodings, known_face_names


def detection_face(frame, face_encoding_know, face_name_know):
    rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame, face_locations)

    face_names_detecte = []
    for face_encoding in face_encodings:

        matches = face_recognition.compare_faces(
            face_encoding_know, face_encoding)

        face_distances = face_recognition.face_distance(
            face_encoding_know, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = face_name_know[best_match_index]
            face_names_detecte.append(name)

    face_locations = np.array(face_locations)
    face_locations = face_locations / 0.25
    return face_locations.astype(int), face_names_detecte


def detection_color(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width, _ = frame.shape
    cx = int(width / 2)
    cy = int(height-100)

    pixel_center = hsv_frame[cy, cx]
    hue_value = pixel_center[0]

    color = "Undefined"
    if hue_value < 5:
        color = "RED"
    elif hue_value < 22:
        color = "ORANGE"
    elif hue_value < 33:
        color = "YELLOW"
    elif hue_value < 78:
        color = "GREEN"
    elif hue_value < 131:
        color = "BLUE"
    elif hue_value < 170:
        color = "VIOLET"
    else:
        color = "RED"

    cv2.putText(frame, "Detection de couleur",(int(width / 2) - 200, 100), 0, 1, (250, 0, 0), 5)
    cv2.circle(frame, (cx, cy), 5, (0, 250, 0), 3)
    print(color)


face_encoding_know = []
face_name_know = []
face_vue = []
detectionEtat = True

face_encoding_know, face_name_know = chargement_images("face_images/")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    if detectionEtat:
        face_locations, face_names = detection_face(small_frame, face_encoding_know, face_name_know)
        
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        if face_names:
            detection_color(frame)


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
