import numpy as np
import cv2, PIL, os
from cv2 import aruco
import sys  
import math
import subprocess
import pickle

def run_calibration():
    calibration_path = '.\新建文件夹\calibration_chessboard.py'
    subprocess.run(["python", calibration_path])
    return None

def get_camera_para(pkl_path,dist_path):
    with open(pkl_path, 'rb') as f:
        # 读取.pkl文件中的数据
        cameraMatrix = pickle.load(f)

    with open(dist_path, 'rb') as f:
        # 读取.pkl文件中的数据
        distCoeffs = pickle.load(f)

    return cameraMatrix, distCoeffs

def ChangeToProperFormat_toReprojection(points_dict):
    points_list = list(points_dict.values())
    points_array = np.array(points_list)
    return points_array

def ReprojectionPoints(marker_size,corners, camera_Matrix, dist_Coeffs):
    square_marker_points = np.array([[-marker_size/2, marker_size/2, 0],
                                 [marker_size/2, marker_size/2, 0],
                                 [marker_size/2, -marker_size/2, 0],
                                 [-marker_size/2, -marker_size/2, 0]], dtype=np.float32)
    retval, rotation_vector, translation_vector = cv2.solvePnP(square_marker_points, corners, camera_Matrix, dist_Coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE )
    reprojected_image_points, _ = cv2.projectPoints(square_marker_points, rotation_vector, translation_vector, camera_Matrix, dist_Coeffs)
    reprojected_image_points = reprojected_image_points.reshape(-1, 2)
    corners = corners.reshape(-1, 2)
    return reprojected_image_points,corners,rotation_vector,translation_vector

def ReprojectionError(after_projection, before_projection):
    reprojectionError = []
    for i in range(len(after_projection)):
        h = np.linalg.norm(after_projection[i]-before_projection[i])
        reprojectionError.append(h)
        return reprojectionError




if __name__=="__main__": 
    pkl_path= "C:/02_FAU/S3/Project_Thesis/Code/Improved_Aruco_Marker/新建文件夹/cameraMatrix.pkl"
    dist_path= "C:/02_FAU/S3/Project_Thesis/Code/Improved_Aruco_Marker/新建文件夹/dist.pkl"
    cameraMatrix, distCoeffs = get_camera_para(pkl_path,dist_path)
    print(cameraMatrix, distCoeffs)