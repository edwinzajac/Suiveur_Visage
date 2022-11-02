from time import sleep, time

print(" Chargement de l'import...")
t0 = time()

import cv2
import mediapipe as mp
from utils_holistic import *


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh_lips_positions , face_oval_landmarks_positions = get_face_mesh_landmarks_positions()

print(" Fin du chargement de l'import", '\n', "Durée :", time()-t0, "seconde(s)")



# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)


#Number of detections
n_detection = 0
n_total_frames = 0

with mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh: #initial confidences = 0.5
  
  while cap.isOpened():
    n_total_frames += 1 
      
    success, image = cap.read()
    
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    brightness(image)
    #image = adjust_gamma(image,2)
    #image = histogram_equalization(image)
    brightness(image)
    results = face_mesh.process(image)
    
            
    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_face_landmarks:
      n_detection += 1 
        
      for face_landmarks in results.multi_face_landmarks:
          
        detect_N_show(image,face_landmarks,mp_face_mesh.FACEMESH_IRISES,face_mesh_lips_positions,face_oval_landmarks_positions)
        
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
            ## A réfléchir si enlever ou non -> Passe le temps de dessin des landmarks de 0.002 à 0.01 sec...
            
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_contours_style())
        
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_IRISES,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_iris_connections_style())
        
    else:
      print('No face detected')
        
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image,1))
    
    if cv2.waitKey(5) & 0xFF == 27: #Echap
      break
      
# Frame number and detection ratio value 
print('\n')
print("Total frames number: ", n_total_frames)    
print("Detection Ratio: ", n_detection/n_total_frames)


cap.release()



## Implémentation du ratio de détection OK

## Instruction "No face detected" OK

## Implémentation du calcul de distance par rapport à l'écran OK

## Calcul de luminance en temps réel OK

## Arriver à récupérer la ligne verticale du face mesh pour récupérer le point du nez

## Algo d'égalisation d'histogrammes - https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

## Courbes de reconnaissance par rapport à la luminance

## Comparaison par rapport au barycentre pour le drône 

## Drône qui se recentre par rapport à l'orientation de la tête