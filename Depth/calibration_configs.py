# filename: calibration_configs.py
import cv2
import numpy as np

#ROS intrinsic/camera matrix 'k d p r' 
left_camera_intr = np.array([[2.298462293636509e+03,0,0],
                            [-8.944966736504700,2.320139612105291e+03,0],
                            [9.865065039905382e+02,4.997233506516963e+02,1]]) #[2.298462293636509e+03,0,0;
                                                                              # -8.944966736504700,2.320139612105291e+03,0;
                                                                              # 9.865065039905382e+02,4.997233506516963e+02,1]

left_camera_matrix = left_camera_intr.T

right_camera_intr = np.array([[2.317317416679335e+03,0,0],
                              [4.109910361617583,2.334101012211686e+03,0],
                              [1.046334289227945e+03,5.279162864798190e+02,1]]) # [2.317317416679335e+03,0,0;
                                                                                # 4.109910361617583,2.334101012211686e+03,0;
                                                                                # 1.046334289227945e+03,5.279162864798190e+02,1]

right_camera_matrix = right_camera_intr.T

# r1、r2-radial,t1、t2-tangential
r1_l = -0.191643909105442  # [-0.191643909105442,0.344583826534612]
r2_l = 0.344583826534612
t1_l = 0.008877641845990    # [0.008877641845990,0.010316186822576]
t2_l = 0.010316186822576
left_distortion = np.zeros((1, 5))
left_distortion[0, 0] = r1_l
left_distortion[0, 1] = r2_l
left_distortion[0, 2] = t1_l
left_distortion[0, 3] = t2_l
left_distortion[0, 4] = 0.0

r1_r = -0.133781068871848 # [-0.133781068871848,0.198469053330639]
r2_r = 0.198469053330639
t1_r = 0.009543459279074  # [0.009543459279074,0.001017015455013]
t2_r = 0.001017015455013
right_distortion = np.zeros((1, 5))
right_distortion[0, 0] = r1_l
right_distortion[0, 1] = r2_l
right_distortion[0, 2] = t1_l
right_distortion[0, 3] = t2_l
right_distortion[0, 4] = 0.0

# Rotation of camera2, essential matrix from MATLAB
R = np.array([[0.009758595263684,-0.256573566261150,0.161351864532109],
              [39.595478183504980,4.445221351617954,6.700032295623598e+02],
              [0.841458323941632,-6.711719247835438e+02,4.403185668510204]]).T
            # [0.009758595263684,-0.256573566261150,0.161351864532109;
            # 39.595478183504980,4.445221351617954,6.700032295623598e+02;
            # 0.841458323941632,-6.711719247835438e+02,4.403185668510204]

# Translation
T = np.array([-6.711868764597448e+02,0.159943741524750,0.257638601184949]) # [-6.711868764597448e+02,0.159943741524750,0.257638601184949]


size = (600, 400)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

# match the rectified and original images
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
# print(R1,R2)
