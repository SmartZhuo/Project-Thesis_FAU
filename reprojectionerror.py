import numpy as np
import cv2, PIL, os
from cv2 import aruco
import sys  
import math
import subprocess
import pickle

def run_calibration():
    """
    The code defines functions to run a calibration script and retrieve camera parameters from pickle
    files.
    :return: The function `get_camera_para` returns the camera matrix (`cameraMatrix`) and distortion
    coefficients (`distCoeffs`) loaded from the pickle files specified by `pkl_path` and `dist_path`
    respectively.
    """
    calibration_path = '.\新建文件夹\calibration_chessboard.py'
    subprocess.run(["python", calibration_path])
    return None

def get_camera_para(pkl_path,dist_path):
    with open(pkl_path, 'rb') as f:
        cameraMatrix = pickle.load(f)

    with open(dist_path, 'rb') as f:
        distCoeffs = pickle.load(f)

    return cameraMatrix, distCoeffs

def ChangeToProperFormat_toReprojection(points_dict):
    """
    The function `ChangeToProperFormat_toReprojection` converts a dictionary of points to a NumPy array.
    
    :param points_dict: It looks like the function `ChangeToProperFormat_toReprojection` is designed to
    convert a dictionary of points into a NumPy array. However, the definition of the `points_dict`
    parameter is missing. Could you please provide an example of how the `points_dict` is structured or
    some
    :return: a NumPy array created from the values of the input dictionary `points_dict`.
    """
    points_list = list(points_dict.values())
    points_array = np.array(points_list)
    return points_array

def ReprojectionPoints(marker_size,corners, camera_Matrix, dist_Coeffs):
    """
    This Python function calculates reprojected image points based on marker size, corners, camera
    matrix, and distortion coefficients.
    
    :param marker_size: Marker size is the size of the marker used in the reprojection process. It is
    typically the size of the square marker in the calibration pattern or object being tracked
    :param corners: The `corners` parameter in the `ReprojectionPoints` function likely represents the
    detected corners of a marker in an image. These corners are typically obtained using a marker
    detection algorithm such as ArUco marker detection
    :param camera_Matrix: The `camera_Matrix` parameter in the `ReprojectionPoints` function represents
    the camera matrix, which contains the intrinsic parameters of the camera. These intrinsic parameters
    include the focal length, optical center, and skew factor. The camera matrix is typically a 3x3
    matrix that is used in camera calibration
    :param dist_Coeffs: The `dist_Coeffs` parameter in the `ReprojectionPoints` function represents the
    distortion coefficients of the camera. These coefficients are used to correct for lens distortion in
    the camera images. They are typically obtained during camera calibration and are used in conjunction
    with the camera matrix to undistort images or
    :return: The function `ReprojectionPoints` returns the reprojected image points, corners, rotation
    vector, and translation vector.
    """
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
    """
    The function calculates the reprojection error between corresponding points in two sets of projected
    coordinates.
    
    :param after_projection: It seems like you were about to provide the definition of the function
    `ReprojectionError`, but the code snippet is incomplete. Could you please provide the definition of
    the `after_projection` parameter so that I can assist you further with completing the function?
    :param before_projection: It seems like you were about to provide some information about the
    `before_projection` parameter, but the message got cut off. Could you please provide more details or
    let me know how I can assist you further with the `before_projection` parameter?
    :return: a list of reprojection errors calculated for each pair of points in the `after_projection`
    and `before_projection` arrays.
    """
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