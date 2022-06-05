import face_recognition
import cv2
import os
import glob
import numpy as np

def chargement_images(chemin):

    known_face_encodings = []
    known_face_names = []

    images_path = glob.glob(os.path.join(chemin, "*.*"))#Get toute les images
    print("{} images trouvés à encoder.".format(len(images_path)))

    for img_path in images_path:
        img = cv2.imread(img_path)#lis l'image
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#converti l'image au format BGR pour face_recognition

        basename = os.path.basename(img_path)#get le nom de l'image
        (filename, ext) = os.path.splitext(basename)

        img_encoding = face_recognition.face_encodings(rgb_img)[0]#encode le visage

        known_face_encodings.append(img_encoding)#Ajoute dans le tableau des visages connus
        known_face_names.append(filename)#Ajoute le nom lié à l'encodage

    print("Encodage fini.")
    return known_face_encodings, known_face_names


def detection_face(frame, face_encoding_know, face_name_know):
    rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converti l'image au format BGR pour face_recognition

    face_locations = face_recognition.face_locations(rgb_small_frame)#Detecte les visages présent sur la frame
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)#Encode les visages

    face_names_detecte = []
    for face_encoding in face_encodings:

        matches = face_recognition.compare_faces(face_encoding_know, face_encoding)#Compare les visages avec ceux connus

        face_distances = face_recognition.face_distance(face_encoding_know, face_encoding)#Get la distance entre le visage detecté et les visages connus
        best_match_index = np.argmin(face_distances)#Get l'id de la meilleur distance entre les visages connus
        if matches[best_match_index]:
            name = face_name_know[best_match_index]#get le nom
            face_names_detecte.append(name)#ajoute le nom au tableau

    face_locations = np.array(face_locations)
    face_locations = face_locations / 0.25
    return face_locations.astype(int), face_names_detecte


def detection_color(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#Convertie la frame en HSV
    height, width, _ = frame.shape #Set le point sur le quel se fait la detection de couleur
    cx = int(width / 2)#position
    cy = int(height-100)

    pixel_center = hsv_frame[cy, cx]
    hue_value = pixel_center[0]#Get l'ID 0 du tableau reçu

    color = "Undefined"
    if hue_value < 5:
        color = "RED"
    elif hue_value < 33:
        color = "YELLOW"
    elif hue_value < 78:
        color = "GREEN"
    elif hue_value < 131:
        color = "BLUE"
    elif hue_value < 170:
        color = "VIOLET"
    else:
        color = "AUTRE"

    cv2.putText(frame, "Detection de couleur",(int(width / 2) - 200, 100), 0, 1, (250, 0, 0), 5)
    cv2.circle(frame, (cx, cy), 5, (0, 250, 0), 3)
    print(color)

#########################################################################################

face_encoding_know = []#Tableau des datas des visages connus
face_name_know = []#Tableau des noms des visages connus
detectionEtat = True

face_encoding_know, face_name_know = chargement_images("face_images/")#Init datas

print("Initlisation de la caméra")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Fin de l'initialisation")

print("Debut reconnaissance")

while True:
    ret, frame = cap.read()#Lis une frame

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)#Resize la frame pour augmenté la vitesse de reconnaissance

    if detectionEtat:
        face_locations, face_names = detection_face(small_frame, face_encoding_know, face_name_know)#Detection de visage
        
        for face_loc, name in zip(face_locations, face_names):#Met un carré sur les visages connus avec le nom de la personne
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        if face_names:#Si visage connus
            detection_color(frame)#Fait la detection de la couleur du haut


    cv2.imshow("Frame", frame)#Affiche l'image

    key = cv2.waitKey(1)
    if key == 27:#Si Echap
        break

cap.release()
cv2.destroyAllWindows()
