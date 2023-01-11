from time import time

print(" Loading imports...")
t0 = time()

from utils_holistic import *

print(" Packages imported", '\n', "Duration :", time()-t0, "second(s)")

# Max limit
max_speed = 20

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

#Choose true if you want to see all the mediapipe landmarks on the image
show_all_landmarks = False

#Number of detections
n_detection = 0
n_total_frames = 0

with mp_face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as face_mesh: #initial confidences = 0.5
  
  while cap.isOpened():
    n_total_frames += 1 
      
    success, image = cap.read()
    
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    
            
    # Draw the face mesh annotations on the image.  
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_face_landmarks:
      n_detection += 1 

      for face_landmarks in results.multi_face_landmarks:
          
        head_barycenter, nose_coord, distance2screen = detect_N_show(image, face_landmarks, mp_face_mesh.FACEMESH_IRISES, show = True)

        mp_drawing.draw_landmarks(
            image = image,
            landmark_list = face_landmarks,
            connections = mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec = None,
            connection_drawing_spec = mp_drawing_styles
            .get_default_face_mesh_contours_style())
            
        if show_all_landmarks:
          mp_drawing.draw_landmarks(
              image = image,
              landmark_list = face_landmarks,
              connections = mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec = None,
              connection_drawing_spec = mp_drawing_styles
              .get_default_face_mesh_tesselation_style())
                     
          mp_drawing.draw_landmarks(
              image = image,
              landmark_list = face_landmarks,
              connections = mp_face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec = None,
              connection_drawing_spec = mp_drawing_styles
              .get_default_face_mesh_iris_connections_style())

      velocity_lr,velocity_fb,velocity_ud,velocity_yaw = drone_velocity(image.shape, head_barycenter, nose_coord, distance2screen)

    else:
      print('No face detected')
      
      velocity_lr, velocity_fb, velocity_ud, velocity_yaw = 0, 0, 0, 0
      
    show_velocities_on_image(image, velocity_lr, velocity_ud, velocity_fb, velocity_yaw, max_speed)
    
    # Check if there is no overspeed
    if not velocities_inrange(velocity_lr, velocity_ud, velocity_fb, velocity_yaw, max_speed):
      print(f'WARNING, max velocity {max_speed} reached')
      
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', image)
        
    if cv2.waitKey(5) & 0xFF == 27: #Echap
      break
      
# Frame number and detection ratio value 
print('\n')
print("Total frames number: ", n_total_frames)    
print("Detection Ratio: ", n_detection/n_total_frames)


cap.release()