import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
ax = plt.axes(projection='3d')

#### Mediapipe utils

def compute_distance(A, B):
    '''
    Compute the distance between two (x,y) vectors
    '''
    
    x = A[0] - B[0]
    y = A[1] - B[1]
    return np.sqrt(x ** 2 + y ** 2)

def get_face_mesh_landmarks_positions():
    '''
        List all the necessary landmarks among all the possible landmarks used by Mediapipe
    '''
    
    face_mesh_lips_landmarks = frozenset([(61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308),
    ])
    
    face_oval_landmarks = frozenset([(10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10)])
    
    return face_mesh_lips_landmarks  , face_oval_landmarks

## Mediapipe landmarks utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh_lips_positions, face_oval_landmarks_positions = get_face_mesh_landmarks_positions()


def brightness(image):
    '''
        Converts a rgb value image into hsv and calculates the v value 
    '''
    
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    v = cv2.split(hsv)[2]
    
    pixel_nb = image.shape[0]*image.shape[1]
    lum_ratio = sum(sum(v))/pixel_nb
    
    print("Luminance =", lum_ratio)

def adjust_gamma(image, gamma=1.0):
    '''
        Applies the gamma correction to the image
    '''
    
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def show_eyes_landmarks(image, face_landmarks, eyes_landmarks_position, show = True):
    '''
        Selects and calculates the barycenter of the eyes from eyes landmarks in mediapipe
    '''
    
    left_eye_x = []
    left_eye_y = []
    
    right_eye_x = []
    right_eye_y = []
    
    i = 0

    for k in eyes_landmarks_position: # k are the indexes of the landmarks and we want only to select the ones from the left eye and then from the right eye
        
        landmark = face_landmarks.landmark
        
        # The two landmarks are the same
        landmark1 = landmark[k[0]]
        # landmark2 = landmark[k[1]]
        
        # Values between 0 and 1 that represent the coordinates of the landmarks
        x1 = landmark1.x
        y1 = landmark1.y
        
        # Coordinates converted to pixel value coordinates
        shape = image.shape 
        relative_x1 = int(x1 * shape[1])
        relative_y1 = int(y1 * shape[0])

        if i in [0,1,5,6]:
            left_eye_x.append(relative_x1)
            left_eye_y.append(relative_y1)
        
        else:
            right_eye_x.append(relative_x1)
            right_eye_y.append(relative_y1)
        
        i += 1
    
    # Computing the barycenters of the left and right eye
    LeftE = (int(np.mean(left_eye_x)), int(np.mean(left_eye_y)))
    RightE = (int(np.mean(right_eye_x)), int(np.mean(right_eye_y)))

    
    if show:
        #Drawing left eye
        cv2.circle(image, LeftE, radius=10, color=(225, 0, 100), thickness=1)
        
        #Drawing right eye
        cv2.circle(image, RightE, radius=10, color=(225, 0, 100), thickness=1)
    
    return LeftE, RightE

def calculate_barycenter(image, face_landmarks, face_landmarks_position, show = True):    
    '''
        Calculates the barycenter considering face landmarks. Needed for lips and head barycenters computation
    '''
    
    ListX = []
    ListY = []
    
    for k in face_landmarks_position:
        
            landmark = face_landmarks.landmark
            
            landmark1 = landmark[k[0]]
            
            x1 = landmark1.x
            y1 = landmark1.y
            
            shape = image.shape 
            relative_x1 = int(x1 * shape[1])
            relative_y1 = int(y1 * shape[0])
            
            ListX.append(relative_x1)
            ListY.append(relative_y1)
    
    Barycenter = (int(np.mean(ListX)), int(np.mean(ListY)))
  
    if show:
        #Drawing lips center
        cv2.circle(image, Barycenter,radius=1, color=(0, 0, 250), thickness=5)
    
    return Barycenter
    
def show_barycenters(image, face_landmarks, show = True):
    '''
        Calculates lips and head barycenters
    '''
    
    Lips_barycenter = calculate_barycenter(image,face_landmarks, face_mesh_lips_positions, show)
    
    Head_barycenter = calculate_barycenter(image,face_landmarks, face_oval_landmarks_positions, show)

    return Lips_barycenter, Head_barycenter
    
def show_nose(image, face_landmarks, show = True):
    '''
        Calculates the nose position considering nose landmark on mediapipe
    '''
    
    nose_landmark = (4, 275)
    landmark = face_landmarks.landmark
    
    landmark1 = landmark[nose_landmark[0]]
    
    x1 = landmark1.x
    y1 = landmark1.y
 
    shape = image.shape 
    relative_x1 = int(x1 * shape[1])
    relative_y1 = int(y1 * shape[0])
    
    if show:
        #Nose landmark drawing
        cv2.circle(image, (relative_x1, relative_y1), radius=1, color=(0, 0, 250), thickness=5)
    
    return (relative_x1,relative_y1)
    
def detect_N_show(image, face_landmarks, eyes_landmarks_position, show = True):
    '''
        Shows the key positions and distance between eyes / distance between the nose and the lips / distance between the face and the screen
    '''
    
    Le, Re = show_eyes_landmarks(image,face_landmarks,eyes_landmarks_position, show)
    Lips_barycenter, Head_barycenter = show_barycenters(image,face_landmarks, show)
    Nose = show_nose(image,face_landmarks, show)
    
    distance_between_eyes = compute_distance(Le, Re)
    distance_nose_lips = compute_distance(Lips_barycenter,Nose)
    distance2screen = int(3700/distance_between_eyes) #Pour essayer, c'est pas du tout exact
    
    print('\n')
    print('Distance between eyes =', distance_between_eyes)
    print('Distance between the nose and the lips =', distance_nose_lips)
    print('Distance to screen =', distance2screen, 'cm')
    
    if show:
        cv2.line(image, Le, Re, color=(0, 0, 255), thickness=2)
        
        cv2.line(image, Lips_barycenter, Head_barycenter, color=(255, 0, 0), thickness=2)
        cv2.line(image, Lips_barycenter, Nose, color=(255, 0, 0), thickness=2)
        cv2.line(image, Nose, Head_barycenter, color=(255, 0, 0), thickness=2)

    return Head_barycenter, Nose, distance2screen

def show_velocities_on_image(image, velocity_lr, velocity_ud, velocity_fb, velocity_yaw, max_speed):
    '''
        Shows the calculated drone velocities on image
    '''
    font                   = cv2.FONT_HERSHEY_PLAIN
    bottomLeftCornerOfText,bottomLeftCornerOfText2,bottomLeftCornerOfText3,bottomLeftCornerOfText4 = (10,200),(10,230),(10,260),(10,290)
    fontScale              = 1
    fontColor              = (0,0,0)
    thickness              = 2
    lineType               = 2
    
    Texte = f"velocity_lr ={velocity_lr}"
    Texte2 = f"velocity_ud ={velocity_ud}"
    Texte3 = f"velocity_fb ={velocity_fb}"
    Texte4 = f"velocity_yaw ={velocity_yaw}"
    
    Txt_speed = ' (max speed)'
    
    if abs(velocity_lr) >= max_speed:
        fontColor = (255,0,0)
        Texte += Txt_speed
    else:
        fontColor = (0,0,0)
        
    if abs(velocity_ud) >= max_speed:
        fontColor2 = (255,0,0)
        Texte2 += Txt_speed
    else:
        fontColor2 = (0,0,0)
        
    if abs(velocity_fb) >= max_speed:
        fontColor3 = (255,0,0)
        Texte3 += Txt_speed
    else:
        fontColor3 = (0,0,0)
        
    if abs(velocity_yaw) >= max_speed:
        fontColor4 = (255,0,0)
        Texte4 += Txt_speed
    else:
        fontColor4 = (0,0,0)
    
    cv2.putText(image,
        Texte, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    cv2.putText(image,
        Texte2, 
        bottomLeftCornerOfText2, 
        font, 
        fontScale,
        fontColor2,
        thickness,
        lineType)
    cv2.putText(image,
        Texte3, 
        bottomLeftCornerOfText3, 
        font, 
        fontScale,
        fontColor3,
        thickness,
        lineType)
    cv2.putText(image,
        Texte4, 
        bottomLeftCornerOfText4, 
        font, 
        fontScale,
        fontColor4,
        thickness,
        lineType)
         

##### Constants

# Drone speed
abs_speed = 25
    
# Distance from screen reference
distance_ref2screen = 60 #cm

# Average displacement
avg_displ = 30
    
    
##### Drone utils

def drone_velocity(image_shape, head_barycenter, nose_coord, distance2screen, simple_move = False, prop_regulator = False):
    '''
        Calculates the drone velocities
        
        If simple_move is True, the drone will rotate instead of moving left and right for centering the head barycenter and will not move forward and backward
        
        If prop_regulator is True, the drone response will follow a proportional regulator rule, else it will be a PID regulation
    '''

    # Coord of the middle
    image_middle = (image_shape[1] // 2, image_shape[0] // 2)
    
    # Calculate velocities
    velocity_lr,velocity_ud = lrud_move(distance2screen, image_middle, head_barycenter, prop_regulator)
    velocity_fb = fb_move(distance2screen, prop_regulator)
    
    if simple_move:
        velocity_yaw = 2*velocity_lr
        velocity_lr = 0
        velocity_fb = 0
        velocity_ud = 2*velocity_ud
        
    else:
        velocity_yaw, add_lr_velocity, add_fb_velocity = circular_move(distance2screen, head_barycenter, nose_coord, prop_regulator)
        velocity_lr += add_lr_velocity
        velocity_fb += add_fb_velocity
        
    return np.rint([velocity_lr, velocity_fb, velocity_ud, velocity_yaw]).astype('int') #converted to int
    
def speed_fct(displacement,distance2screen, regulateur_prop = False):
    '''
        Calculate the absolute speed of drone depending on displacement & distance to screen
    '''
    
    #Exp factor
    A = 1/(np.exp(1) - 1)
    
    scale_through_distance = distance2screen/distance_ref2screen
    
    speed_value = scale_through_distance * abs_speed 
    
    if regulateur_prop:
        res = speed_value*displacement
        
    else:
        if displacement >= 0:
            res =  A*(np.exp(displacement/avg_displ)-1) * scale_through_distance * abs_speed
        else:
            res = - A*(np.exp(-displacement/avg_displ)-1) * scale_through_distance * abs_speed
        
    return res

def lrud_move(distance2screen, destination_coord, initial_coord, regulateur_prop = False):
    
    lr_displacement = (destination_coord[0] - initial_coord[0])/40
    ud_displacement = (destination_coord[1] - initial_coord[1])/40
    
    velocity_lr = 4*speed_fct(lr_displacement,distance2screen, regulateur_prop)
    velocity_ud = 6*speed_fct(ud_displacement,distance2screen, regulateur_prop)
    
    return velocity_lr,velocity_ud
    
def fb_move(distance2screen, prop_regulator = False):
    
    fb_displacement = (distance2screen - distance_ref2screen)/5
    
    if fb_displacement < 0:
        fb_displacement = -fb_displacement**2 # if the drone is too close, increase the speed to make it move back
    
    return speed_fct(fb_displacement, distance2screen, prop_regulator)
    
def yaw_move(distance2screen, head_barycenter, nose_coord, prop_regulator = False):
    # Comparing nose and barycenter position
    displacement = (head_barycenter[0] - nose_coord[0])/10
    return 4*speed_fct(displacement, distance2screen, prop_regulator)

def circular_move(distance2screen, head_barycenter, nose_coord, prop_regulator = False):
    '''
        Calculates the velocities for a circular move around the head
    '''
    velocity_yaw = - yaw_move(distance2screen, head_barycenter, nose_coord, prop_regulator) # "- because it must be opposite to the rotation of the head"
    
    step_duration = 0.05 #average time of the main loop
    
    theta = velocity_yaw / distance2screen * step_duration
    
    add_lr_velocity = -2*velocity_yaw * np.cos(theta)
    add_fb_velocity = 2/3*velocity_yaw * np.sin(theta)
    
    #Values equilibration
    velocity_yaw = 3/2 * velocity_yaw
    
    return velocity_yaw, add_lr_velocity, add_fb_velocity
    
def velocities_inrange(velocity_lr, velocity_ud, velocity_fb, velocity_yaw, max_speed):
    '''
        Return True if the drone velocities are in the acceptance range [0,max_speed]
    '''

    print("Velocity_lr", velocity_lr)
    print("Velocity_ud", velocity_ud)
    print("Velocity_fb", velocity_fb)
    print("Velocity_yaw", velocity_yaw)
    
    return abs(max(velocity_lr,velocity_ud,velocity_fb,velocity_yaw)) < max_speed
