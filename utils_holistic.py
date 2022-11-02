import cv2
import numpy as np
import mediapipe as mp
from time import sleep


def compute_distance(A, B):
    x = A[0] - B[0]
    y = A[1] - B[1]
    return np.sqrt(x ** 2 + y ** 2)

def get_face_mesh_landmarks_positions():
    face_mesh_lips = frozenset([(61, 146),
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
    
    face_oval = frozenset([(10, 338),
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
    
    return face_mesh_lips  , face_oval

def brightness(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    v = cv2.split(hsv)[2]
    
    pixel_nb = image.shape[0]*image.shape[1]
    lum_ratio = sum(sum(v))/pixel_nb
    
    print("Luminance =", lum_ratio)

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def histogram_equalization(image):
    equ = cv2.equalizeHist(image)
    res = np.hstack((image, equ))  # stacking images side-by-side
    return res

def show_eyes_landmarks(image,face_landmarks,eyes_landmarks_position):
    left_eye_x = []
    left_eye_y = []
    
    right_eye_x = []
    right_eye_y = []
    
    i = 0

    for k in eyes_landmarks_position:
        
        landmark = face_landmarks.landmark
        
        #Bizarrement le landmark1 suffit...
        landmark1 = landmark[k[0]]
        #landmark2 = landmark[k[1]]
        
        x1 = landmark1.x
        y1 = landmark1.y
        
        #x2 = landmark2.x
        #y2 = landmark2.y
        
        shape = image.shape 
        relative_x1 = int(x1 * shape[1])
        relative_y1 = int(y1 * shape[0])
        
        #relative_x2 = int(x2 * shape[1])
        #relative_y2 = int(y2 * shape[0]) 
        if i in [0,1,5,6]:
            left_eye_x.append(relative_x1)
            left_eye_y.append(relative_y1)
        
        else:
            right_eye_x.append(relative_x1)
            right_eye_y.append(relative_y1)
        #relative_x.append(relative_x2)
        #relative_y.append(relative_y2)
        
        #Draw iris landmarks
        #cv2.circle(image, (relative_x1, relative_y1), radius=10, color=(225, 0, 100), thickness=1)
        #cv2.circle(image, (relative_x2, relative_y2), radius=10, color=(225, 0, 100), thickness=1)
        
        i += 1
    
    LeftE = (int(np.mean(left_eye_x)), int(np.mean(left_eye_y)))
    RightE = (int(np.mean(right_eye_x)), int(np.mean(right_eye_y)))

    
     
    #Drawing left eye
    cv2.circle(image, LeftE, radius=10, color=(225, 0, 100), thickness=1)
    
    #Drawing right eye
    cv2.circle(image, RightE, radius=10, color=(225, 0, 100), thickness=1)
    
    return LeftE, RightE

def calculate_barycenter(image,face_landmarks,face_landmarks_position):
    
    ListX = []
    ListY = []
    
    for k in face_landmarks_position:
        
            landmark = face_landmarks.landmark
            
            #Bizarrement le landmark1 suffit...
            landmark1 = landmark[k[0]]
            landmark2 = landmark[k[1]]
            
            x1 = landmark1.x
            y1 = landmark1.y
            
            #x2 = landmark2.x
            #y2 = landmark2.y
            
            shape = image.shape 
            relative_x1 = int(x1 * shape[1])
            relative_y1 = int(y1 * shape[0])
            
            #relative_x2 = int(x2 * shape[1])
            #relative_y2 = int(y2 * shape[0])
            
            ListX.append(relative_x1)
            ListY.append(relative_y1)
    
            #cv2.circle(image, (relative_x1, relative_y1), radius=10, color=(225, 0, 100), thickness=1)
            #cv2.circle(image, (relative_x2, relative_y2), radius=10, color=(225, 0, 100), thickness=1)

    Barycenter = (int(np.mean(ListX)), int(np.mean(ListY)))

    #Drawing lips center
    cv2.circle(image, Barycenter, radius=10, color=(225, 0, 100), thickness=1)
    
    return Barycenter
    
def show_barycenters(image,face_landmarks,lips_landmarks_position,face_oval_landmarks_position):
    
    Lips_barycenter = calculate_barycenter(image,face_landmarks,lips_landmarks_position)
    
    Head_barycenter = calculate_barycenter(image,face_landmarks,face_oval_landmarks_position)

    return Lips_barycenter, Head_barycenter
    
def detect_N_show(image,face_landmarks,eyes_landmarks_position,lips_landmarks_position,face_oval_landmarks_position):
    
    Le, Re = show_eyes_landmarks(image,face_landmarks,eyes_landmarks_position)
    Lips_barycenter, Head_barycenter = show_barycenters(image,face_landmarks,lips_landmarks_position,face_oval_landmarks_position)
    
    distance_between_eyes = compute_distance(Le, Re)
    distance_between_barycenters = compute_distance(Lips_barycenter,Head_barycenter)
    distance2screen = 30*110/distance_between_eyes #Pour essayer, c'est pas du tout exact
    
    print('\n')
    print('Distance between eyes =', distance_between_eyes)
    print('Distance between the barycenter and the lips =', distance_between_barycenters)
    print('Distance to screen =', distance2screen, 'cm')
    
    cv2.line(image, Le, Re, color=(0, 0, 255), thickness=2)
    cv2.line(image, Lips_barycenter, Head_barycenter, color=(255, 0, 0), thickness=2)
    
    