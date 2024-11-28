###############################
# Importing necessary modules #
###############################
import cv2
import mediapipe as mp
import numpy as np

#######################################
# Does your screen has the 125% zoom? #
#######################################
resposta= 0
if resposta==1:
  n_125=1.25
else:
  n_125=1

#####################
# Camera Parameters #
#####################
# Calibrated
cam_matrix = np.array([[663.43273799, 0, 304.35476967],
                       [0, 663.43273799, 235.45448301],
                       [0, 0, 1]], dtype="double")
distortion_matrix = np.array([-0.19861746, 0.60201285, -0.00317874, -0.00460291])
# # Approximation
# img_w, img_h = (640,480)
# focal_length = 1 * img_w
# cam_matrix = np.array([[focal_length, 0, img_h / 2],
#                        [0, focal_length, img_w / 2],
#                        [0, 0, 1]], dtype="double")
# distortion_matrix = np.zeros((4, 1))

########################
# Computing Homography #
########################
# Rectangle Coordinates (Computer Screen - origin)
rect_pts = np.array([[0, 0], [1920, 0], [1920, 816], [0, 816]], dtype=np.float32)

# Trapezoid Coordinates (Projected Screen)
trap_pts = np.array([[53, 0], [1908, 0], [1908, 775], [39, 808]], dtype=np.float32)

# Computing homography matrix
H, _ = cv2.findHomography(rect_pts, trap_pts)
# print(H)

########################################
# Transforming the background template #
########################################
# Uploading image
img_check = cv2.imread('fons_pantalla_pc4.png')
img_check = cv2.resize(img_check,(round(1920/n_125),round(816/n_125)))
size_v, size_h, _ = img_check.shape

# Transforming image
img_trap = cv2.warpPerspective(img_check, H, (1920, 816))

#####################################
# Defining parameters and variables #
#####################################
# Detection limit angles (yaw - pitch)
n_interval_h_world, n_interval_v_world = 35, 20

# Landmark indices to be compared with the model
index_shortproject = [4, 175, 33, 263, 76, 306]

# 3d Head generic model points
model_points = np.array([
  (0.0, 0.0, 0.0),            # Nose tip
  (0.0, -330.0, -65.0),       # Chin
  (-225.0, 170.0, -135.0),    # Left eye left corner
  (225.0, 170.0, -135.0),     # Right eye right corner
  (-150.0, -150.0, -125.0),   # Left Mouth corner
  (150.0, -150.0, -125.0)     # Right mouth corner
])

# Size and position of the text box (grey rectangle)
text_box_size = (160, 210)
text_box_position = (475, 17)

# Size and position of the normalized rectangle (inside the grey rectangle)
text_rect_size = (130, 80)
text_rect_position = (490, 130)

# Printing Coordinates for Computer Screen window
text_position_pantalla_PC = (img_check.shape[1] - 400, 30)
angle_position_pantalla_PC = (img_check.shape[1] - 1700, 30)
stat_vel = 'VELOCITY: Low'

# Attractors definiton (values in degrees - (yaw, pitch))
centre_attractor = [(7, 5), (-7, 0), (23, 10), (20, -2), (-16, 9.5), (-15.6, -11.6)]
margin_attractor = (4, 3)

# Initializing variables
plotx, ploty = 0, 0               # angles to visually display the real value
plotx_dot, ploty_dot = 0, 0       # angles to apply to the user-driven dot position
n_high,n_low = 0, 0               # number of frames at low/high speed
vel_th = 6                        # threshold of angle difference between frames

###############################
# Head Pose Estimation System #
###############################
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.85) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()

    # Checking if frame successfully retrieved
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # Image dimensions
    img_h, img_w, img_c = image.shape
    # print(img_w, img_h)

    # Saving previous values of the angles
    plotx_ant, ploty_ant = plotx, ploty

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)      # Flip the image horizontally for a selfie-view display.
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:

        # Filtering landmarks to only show the specified ones
        filtered_landmarks = [face_landmarks.landmark[i] for i in index_shortproject]

        # Get 2d Coord as an array
        image_points = np.array([(face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h) for i in index_shortproject], dtype="double")

        # Draw the landmarks on the image.
        for i, landmark in enumerate(filtered_landmarks):
            x,y = int(landmark.x * img_w),int(landmark.y * img_h)
            color = (0, 0, 255)
            cv2.circle(image, (x, y), 2, color, -1)

        # Solving the PnP problem
        success, rotation_vec, translation_vec = cv2.solvePnP(model_points, image_points, cam_matrix, distortion_matrix, cv2.SOLVEPNP_ITERATIVE)

        # Computing head rotation
        rmat, jac = cv2.Rodrigues(rotation_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        pitch = (angles[0]+180)
        yaw = -angles[1]
        roll = angles[2]
        if 300 <= pitch <= 360:
          pitch = pitch - 360
        pitch = -pitch
        plotx, ploty = yaw, pitch

        # Blocking the value of the variable if exceeds the world limit
        if abs(plotx) > n_interval_h_world:
          plotx = plotx_ant
        if abs(ploty) > n_interval_v_world:
          ploty = ploty_ant

        # Filtering
        if np.sqrt(abs(plotx - plotx_ant) ** 2 + abs(ploty - ploty_ant) ** 2) > vel_th:
          beta = [0.3, 0.3]
          n_high += 1
          n_low = 0
          if n_high >= 0:
            stat_vel = 'VELOCITY: High'
            print('VELOCITY: High')
        else:
          beta = [0.05, 0.05]
          n_high = 0
          n_low += 1
          if n_low >= 5:
            stat_vel = 'VELOCITY: Low'
            print('VELOCITY: Low')
        plotx = beta[0] * plotx + (1-beta[0]) * plotx_ant
        ploty = beta[1] * ploty + (1-beta[1]) * ploty_ant
        plotx_dot = plotx
        ploty_dot = ploty

        # Applying the attractor effect
        for centre in centre_attractor:
          centre_x, centre_y = centre
          if centre_x - margin_attractor[0] < plotx < centre_x + margin_attractor[0] and centre_y - margin_attractor[1] < ploty < centre_y + margin_attractor[1]:
            plotx_dot = centre_x
            ploty_dot = centre_y
            break

    ###############################################################
    # Plotting rectangles, attractors, dots and presenting values #
    ###############################################################
    # General grey rectangle with text (Camera window)
    text_image = np.ones((text_box_size[1], text_box_size[0], 3), dtype=np.uint8) * 185
    cv2.rectangle(text_image, (0, 0), (text_box_size[0] - 1, text_box_size[1] - 1), (0, 0, 0), 2)

    # Specifying rectangle coordinates (Camera window)
    start_x = text_box_position[0]
    end_x = start_x + text_box_size[0]
    start_y = text_box_position[1]
    end_y = start_y + text_box_size[1]
    image[start_y:end_y, start_x:end_x] = text_image

    # Plotting YAW and PITCH values (Camera window)
    cv2.putText(image, "YAW: " + str(np.round(plotx, 1)), (495, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(image, "PTH: " + str(np.round(ploty, 1)), (495, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Normalized rectangle for the red dot (Camera window)
    text_rect = np.ones((text_rect_size[1], text_rect_size[0], 3), dtype=np.uint8) * 255
    cv2.rectangle(text_rect, (0, 0), (text_rect_size[0] - 1, text_rect_size[1] - 1), (0, 0, 0), 2)

    # Specifying normalized rectangle coordinates (Camera window)
    start_x = text_rect_position[0]
    end_x = start_x + text_rect_size[0]
    start_y = text_rect_position[1]
    end_y = start_y + text_rect_size[1]
    image[start_y:end_y, start_x:end_x] = text_rect

    # Central cross for the normalized rectangle (Camera window)
    rec_petit_center_h = round(text_rect_position[0] + text_rect_size[0] / 2)
    rec_petit_center_v = round(text_rect_position[1] + text_rect_size[1] / 2)
    rec_petit_center = (rec_petit_center_h, rec_petit_center_v)
    t = 10
    cv2.line(image, (rec_petit_center_h - t, rec_petit_center_v), (rec_petit_center_h + t, rec_petit_center_v), (0, 0, 0), 1)
    cv2.line(image, (rec_petit_center_h, rec_petit_center_v - t), (rec_petit_center_h, rec_petit_center_v + t), (0, 0, 0), 1)

    # Attractors in the normalized rectangle (Camera widow)
    for centre in centre_attractor:
      centre_x, centre_y = centre
      atractor_h = round((text_rect_size[0] / (2 * n_interval_h_world)) * centre_x + text_rect_size[0] / 2 + text_rect_position[0])
      atractor_v = round(-(text_rect_size[1] / (2 * n_interval_v_world)) * centre_y + text_rect_size[1] / 2 + text_rect_position[1])
      atractor = (atractor_h, atractor_v)
      cv2.circle(image, atractor, 5, (255, 0, 0), -1)

    # User-driven red dot for the normalized rectangle (Camera window)
    rec_petit_h = round((text_rect_size[0] / (2 * n_interval_h_world)) * plotx_dot + text_rect_size[0] / 2 + text_rect_position[0])
    rec_petit_v = round(-(text_rect_size[1] / (2 * n_interval_v_world)) * ploty_dot + text_rect_size[1] / 2 + text_rect_position[1])
    rec_petit = (rec_petit_h, rec_petit_v)
    cv2.circle(image, rec_petit, 3, (0, 0, 255), -1)

    # User-driven red dot (Computer Screen window)
    rec_petit_h = round((size_h / (2 * n_interval_h_world)) * plotx_dot + size_h / 2)
    rec_petit_v = round(-(size_v / (2 * n_interval_v_world)) * ploty_dot + size_v / 2)
    rec_petit = np.array([[rec_petit_h], [rec_petit_v], [1]])

    # User-driven red dot transformed by the homography (Projector window)
    rec_petit_homografiat = np.dot(H, rec_petit)
    punt_x_norm = int(round(rec_petit_homografiat[0, 0] / rec_petit_homografiat[2, 0]))
    punt_y_norm = int(round(rec_petit_homografiat[1, 0] / rec_petit_homografiat[2, 0]))

    # Creating images for Computer screen and Projector screen
    image2 = img_check.copy()
    image3 = img_trap.copy()
    for centre in centre_attractor:
      centre_x, centre_y = centre
      atractor_h = round((size_h / (2 * n_interval_h_world)) * centre_x + size_h / 2)
      atractor_v = round(-(size_v / (2 * n_interval_v_world)) * centre_y + size_v / 2)
      atractor = (atractor_h, atractor_v)
      cv2.circle(image2, atractor, 25, (255, 0, 0), -1)
      atractor = np.array([[atractor_h], [atractor_v], [1]])
      atractor_homografiat = np.dot(H, atractor)
      atractor_x_norm = int(round(atractor_homografiat[0, 0] / atractor_homografiat[2, 0]))
      atractor_y_norm = int(round(atractor_homografiat[1, 0] / atractor_homografiat[2, 0]))
      cv2.circle(image3, (atractor_x_norm, atractor_y_norm), 25, (255, 0, 0), -1)
    cv2.circle(image2, (rec_petit_h, rec_petit_v), 15, (0, 0, 255), -1)
    cv2.putText(image2, stat_vel, text_position_pantalla_PC, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(image2, "YAW: " + str(round(plotx, 1)) + "    PITCH: " + str(round(ploty, 1)), angle_position_pantalla_PC, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.circle(image3, (punt_x_norm, punt_y_norm), 15, (0, 0, 255), -1)

    # Presenting the images in different windows
    cv2.imshow('Camera Window', image)
    cv2.imshow('Computer Screen Window', cv2.resize(image2, (round(1920 / (n_125)), round(816 / (n_125)))))
    cv2.imshow('Projector Window', cv2.resize(image3, (1920, 816)))

    # Enabling to stop program by pressing 'q'
    if cv2.waitKey(5) == ord('q'):
      break

cap.release()
