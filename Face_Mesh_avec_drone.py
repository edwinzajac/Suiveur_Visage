from time import time

print(" Loading imports and connection...")
t0 = time()

import cv2
from utils_holistic import *
from djitellopy import Tello

# Drone speed limit
max_speed = 50

# Load cascade classifier 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Tello setup
tello = Tello()
tello.connect()
print(" Tello battery =", tello.get_battery())
tello.set_speed(abs_speed) #abs_speed = 25
tello.streamon()
tello.takeoff()

print(" Packages imported and drone connected", '\n', "Duration :", time()-t0, "second(s)")
print(" Drone is taking off ...")

# Init object to read video frames from Tello
frame_read = tello.get_frame_read()
image_shape = frame_read.frame.shape[:2]
video_capture = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, image_shape)

# Choose true if you want to see all the mediapipe landmarks on the image
show_all_landmarks = False

# Number of detections
n_detection = 0
n_total_frames = 0

with mp_face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as face_mesh: #initial confidences = 0.5
  
  while True:
    n_total_frames += 1 
      
    # Obtaining video
    image = np.fliplr(frame_read.frame)

    # Face detection
    results = face_mesh.process(image)
    
    # Draw the face mesh annotations on the image.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      n_detection += 1 
        
      for face_landmarks in results.multi_face_landmarks:
        head_barycenter, nose_coord, distance2screen = detect_N_show(image,face_landmarks,mp_face_mesh.FACEMESH_IRISES, show = True)

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

      # Update velocities
      velocity_lr,velocity_fb,velocity_ud,velocity_yaw = drone_velocity(image.shape, head_barycenter, nose_coord, distance2screen, rotation_only = True)
        
    else:
      print('No face detected')
      velocity_lr, velocity_fb, velocity_ud, velocity_yaw = 0, 0, 0, 0

    # Check if there is no overspeed
    if not velocities_inrange(velocity_lr, velocity_ud, velocity_fb, velocity_yaw, max_speed):
      print(f'WARNING, max velocity {max_speed} reached')
      
      velocity_lr = min(velocity_lr, max_speed)
      velocity_fb = min(velocity_fb, max_speed)
      velocity_ud = min(velocity_ud, max_speed)
      velocity_yaw = min(velocity_yaw, max_speed)
    
    tello.send_rc_control(velocity_lr, velocity_fb, velocity_ud, velocity_yaw)
    
    #Show velocities on image
    show_velocities_on_image(image, velocity_lr, velocity_ud, velocity_fb, velocity_yaw, max_speed)
    
    # Saving video
    video_capture.write(image)
    print("Frame ajoutée à la vidéo :")

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', image)
    
    if cv2.waitKey(5) & 0xFF == 27: #Echap
      break

# Stopping video
video_capture.release()

# Closes all the frames
cv2.destroyAllWindows()

# Frame number and detection ratio value 
print('\n')
print("Total frames number: ", n_total_frames)    
print("Detection Ratio: ", n_detection/n_total_frames)

print(" Tello battery =", tello.get_battery())

# Landing
tello.land()
tello.end()

## Algo d'égalisation d'histogrammes - https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

## Courbes de reconnaissance par rapport à la luminance

