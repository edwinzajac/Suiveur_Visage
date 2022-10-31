import cv2
import mediapipe as mp
from utils_holistic import *

# à faire: trouver un critère qui définit de façon robuste l'éloignement de la personne par rapport
# à la caméra.
# piste: extraire les caractéristiques du cadre par exemple.

# ce serait peut-être mieux de travailler à partir des landmarks de holistic ('face mesh'), il y a mions de faux négatifs
# quand on tourne la tête, et on pourrait avoir l'équivalent d'une boîte englobante.

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #image = adjust_gamma(image,0.1)


        results = holistic.process(image)

        Coord_Right_Eye, Coord_Left_Eye, Coord_Nose, Coord_Mouth, Eyes_distance, Mouth_nose_distance, B_centre = detect(
            image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display
        
        
        show_landmarks(Coord_Right_Eye, Coord_Left_Eye, Coord_Nose, Coord_Mouth, Eyes_distance, Mouth_nose_distance, B_centre,image)

        cv2.imshow('MediaPipe Holistic and points', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:#echap
            break
cap.release()
