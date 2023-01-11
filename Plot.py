from utils_holistic import *

def show_response(simple_move, prop_regulator):
    image_shape = (480, 640, 3) #webcam shape
    Nose = [(i,0) for i in range(image_shape[0])]
    Nose = [(240,320)]
    Head = [(i,j) for i in range(image_shape[0]) for j in range(image_shape[1])]
    d2s = 50
    
    
    V_lr = []
    V_fb = []
    V_ud = []
    V_yaw = []
    C = []
    
    for nose in Nose:
        for head in Head:
            velocity_lr, velocity_fb, velocity_ud, velocity_yaw = drone_velocity(image_shape, head, nose, d2s, simple_move, prop_regulator)
            C.append((nose[0],head[0],head[1]))
            V_lr.append(velocity_lr)
            V_fb.append(velocity_fb)
            V_ud.append(velocity_ud)
            V_yaw.append(velocity_yaw)
    
    return  V_lr, V_fb, V_ud, V_yaw, C


# Calculate velocities
V_lr, V_fb, V_ud, V_yaw, C = show_response(simple_move = False, prop_regulator = False)

#Show velocities
nose, headx, heady = zip(*C) # [a,b,c] -> [a], [b], [c]


ax.plot3D(headx, heady, V_lr, label = 'lr' )

ax.plot3D(headx, heady, V_fb, label = 'fb')

ax.plot3D(headx, heady, V_ud, label = 'ud')

ax.plot3D(headx, heady, V_yaw, label = 'yaw')

ax.set_xlabel('Head_X')
ax.set_ylabel('Head_Y')
ax.set_zlabel('Velocities')

plt.legend()
plt.show()