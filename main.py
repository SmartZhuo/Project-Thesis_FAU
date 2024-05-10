
import numpy as np
import cv2, PIL, os
from cv2 import aruco
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon,Point
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
import sys  
import math
import subprocess
from Subpix import get_subpix_corners
import reprojectionerror as rp


class Image:
    def __init__(self, id: int, image_path: str, aruco_dict):
        self.id = id
        self.image_path = image_path
        self.aruco_dict = aruco_dict


    def detect_markers(self):
        """
        The function `detect_markers` reads an image, converts it to grayscale, detects ArUco markers, and
        returns the corners, IDs, rejected image points, and the original image.
        :return: The `detect_markers` method returns four values: `corners`, `ids`, `rejectedImgPoints`, and
        `image`.
        """
        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=parameters)
        return corners, ids, rejectedImgPoints,image
    
    def calculate_marker_area(self,corners):
        """
        The function calculates the area of a marker based on the coordinates of its corners.
        
        :param corners: It looks like the code you provided is a function that calculates the area of a
        marker based on its corner coordinates. However, you have not provided the actual values for the
        `corners` parameter. If you provide me with the values for the `corners` parameter, I can help you
        calculate
        :return: The function `calculate_marker_area` is returning the area of a polygon formed by the
        corners provided as input.
        """
        x1 = int(corners[0][0][0][0])
        y1 = int(corners[0][0][0][1])

        x2 = int(corners[0][0][1][0])
        y2 = int(corners[0][0][1][1])

        x3 = int(corners[0][0][2][0])
        y3 = int(corners[0][0][2][1])

        x4 = int(corners[0][0][3][0])
        y4 = int(corners[0][0][3][1])
        vertices = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]  
        polygon = Polygon(vertices)  
        marker_area = polygon.area  
        return marker_area
    
    def Get_quadrilaternal(self,corners):
        """
        The function `Get_quadrilateral` takes a list of corner points, converts them to a numpy array, and
        returns the first point as an integer.
        
        :param corners: It looks like the `corners` parameter is a list of corner points of a quadrilateral.
        The function `Get_quadrilateral` takes this list of corner points, converts it to a numpy array, and
        then returns the first corner point as an integer
        :return: The code is returning the first point of the quadrilateral, which is the point at index 0
        of the quadrilateral_points array.
        """
        corners1=np.array(corners[0]) 
        quadrilateral_points = corners1.astype(int)
        return quadrilateral_points[0]
    

    """
    The function `ExpandArea` takes a scaling factor `K` and a set of quadrilateral points, calculates
    the expanded points based on the scaling factor, and returns the expanded points along with the
    width and height of the quadrilateral.
    
    :param K: The parameter `K` in the `ExpandArea` function seems to be a scaling factor that
    determines how much the quadrilateral should be expanded. It is used to scale the vectors from the
    center of the quadrilateral to each of its points in order to expand the shape
    :param quadrilateral_points: The `quadrilateral_points` parameter in the `ExpandArea` function
    represents the coordinates of the four points that define a quadrilateral shape. These points are
    used to calculate the center of the quadrilateral, expand the area by a factor of `K`, and determine
    the width and height of the
    :return: The function `ExpandArea` returns three values: 
    1. `expanded_points`: The coordinates of the quadrilateral points after expanding by a factor of
    `K`.
    2. `w`: The width of the expanded quadrilateral.
    3. `h`: The height of the expanded quadrilateral.
    """
    def ExpandArea(self,K,quadrilateral_points):   
        center = np.mean(quadrilateral_points, axis=0)  
        vectors = quadrilateral_points - center  
        expanded_vectors = vectors * K   
        expanded_points = expanded_vectors + center  
        left_top_point = min(quadrilateral_points, key=lambda point: (point[0], point[1]))
        right_bottom_point = max(quadrilateral_points, key=lambda point: (point[0], point[1]))
        w = abs(right_bottom_point[0] - left_top_point[0])
        h = abs(right_bottom_point[1] - left_top_point[1])
        return expanded_points,w,h
    
    def Get_ROI(self,expanded_points,image):
        """
        This function calculates the Region of Interest (ROI) in an image based on given expanded points.
        
        :param expanded_points: It looks like the description of the `expanded_points` parameter is
        missing. Could you please provide more information about what `expanded_points` represents or how
        it is structured? This will help me understand the context in which the function `Get_ROI` is
        being used and provide you with a more accurate
        :param image: The function `Get_ROI` takes in two parameters: `expanded_points` and `image`
        :return: a region of interest (ROI) from the input image based on the minimum and maximum x and y
        coordinates of the expanded points provided.
        """
        x_coords = expanded_points[:, 0]  
        y_coords = expanded_points[:, 1]  
        x_min = int(x_coords.min()) 
        x_max = int(x_coords.max())
        y_min = int(y_coords.min())
        y_max = int(y_coords.max())
        roi = image[y_min : y_max,x_min : x_max]
        return roi

    def Ellipse_detection(self,roi,w,h):
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
        parameters =  aruco.DetectorParameters_create()
        roi_corners, roi_ids, roi_rejectedImgPoints = aruco.detectMarkers(roi_gray, aruco_dict, parameters=parameters)
        roi_blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        roi_edges = cv2.Canny(roi_blurred, 50, 150)  
        roi_corners1=np.array(roi_corners[0]) 
     # The above code snippet is performing the following tasks:
        roi_corners_array = roi_corners1.astype(int)
        contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        valid_contours = []
        for contour in contours:
            if len(contour) >= 5:  
                ellipse = cv2.fitEllipse(contour)  
                print(f'ellipse: {ellipse}')
            else: print('No Epplise found')
            #   椭圆的中心点 (x, y)、椭圆次轴和主轴的半径 (b, a)，以及椭圆的旋转角度 angle
            (x,y),(b,a),angle = ellipse
            threshold_bmin = int(np.sqrt(max(w,h)))
            threshold_amax = max(w,h)
            # center of Ellipse not in the marker area
            polygon = Polygon(roi_corners_array[0])
            # center of the polygon centroid.x and centroid.y
            centroid = polygon.centroid
            roi_center = np.array([centroid.x, centroid.y])

            if int(a) <= threshold_amax  and int(b)>= threshold_bmin:
                #print(f'polygon.contains(Point(x, y): {polygon.contains(Point(x, y))}')
                if not polygon.contains(Point(x, y)):
                    valid_contours.append(contour)  
            # draw ellipse on ROI 
        # find the closest 4 ellipses to the center of the marker
        valid_contours.sort(key=lambda contour: np.linalg.norm(roi_center - cv2.minEnclosingCircle(contour)[0]))
        roi_ellipse = valid_contours[:4]
        ellipse_centers = [] 
        for contour in roi_ellipse:  
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (x, y), (b, a), angle = ellipse
                centroid_x, centroid_y = int(x), int(y)
                ellipse_centers.append((centroid_x, centroid_y)) 
                #print(f"Ellipse Center is：({centroid_x}, {centroid_y})")
                
        return ellipse_centers,roi_corners
    
    def getCornerinOrder(self,ellipse_centers):
        """
        This Python function takes a list of ellipse centers and returns the coordinates of the corners
        (top-left, top-right, bottom-left, bottom-right) based on the average position of the centers.
        
        :param ellipse_centers: The `getCornerinOrder` function takes a list of ellipse centers as input
        and determines the positions of the corners of a bounding box around these ellipses. The function
        calculates the average x and y coordinates of all the ellipse centers and then assigns each ellipse
        center to one of the four corners based
        :return: The `getCornerinOrder` function returns a dictionary `positions` containing the
        coordinates of the corners of a shape based on the input `ellipse_centers`. The keys of the
        dictionary represent the corner positions ('top-left', 'top-right', 'bottom-left', 'bottom-right'),
        and the values are the corresponding coordinates of those corners.
        """
        positions = {  
        'top-left': None,  
        'top-right': None,  
        'bottom-left': None,  
        'bottom-right': None  
        }  

        x_coordinates = [center[0] for center in ellipse_centers]  
        y_coordinates = [center[1] for center in ellipse_centers]  
 
        average_x = sum(x_coordinates) / len(x_coordinates)  
        average_y = sum(y_coordinates) / len(y_coordinates)  
        #print(average_x,average_y)
        # Iterate over the points and find their corresponding positions  
        for point in ellipse_centers:  
            x, y = point  
            if positions['top-left'] is None and x <= average_x and y <= average_y:  
                positions['top-left'] = point  
            elif positions['top-right'] is None and x >= average_x and y <= average_y: 
                positions['top-right'] = point  
            elif positions['bottom-right'] is None and x >= average_x and y >= average_y: 
                positions['bottom-right'] = point  
            elif positions['bottom-left'] is None and x<= average_x and y >= average_y: 
                positions['bottom-left'] = point  
    
        return positions
    
    def nearest_point_on_line(self,x1, y1, x2, y2, x0, y0):
        # get the line param
        A = y2 - y1
        B = x1 - x2
        C = x2*y1 - x1*y2
        # distance
        distance = abs(A*x0 + B*y0 + C) / ((A**2 + B**2)**0.5)
        # closest point on the line
        x = (B * (B * x0 - A * y0) - A * C) / (A**2 + B**2)
        y = (A * (y0 - B * x0) - B * C) / (A**2 + B**2)

        return (x, y), distance
    

    
        """
        The function `line_fitting` fits lines between specified points using an objective function and
        optimization to find the optimal parameter.
        
        :param roi_corners: The `roi_corners` parameter seems to represent the corners of a region of
        interest (ROI). The function `line_fitting` takes this parameter along with
        `ellipse_centers_inorder` to fit lines between specific pairs of corners and ellipse centers. The
        objective is to find the optimal parameter `t`
        :param ellipse_centers_inorder: Ellipse_centers_inorder is a dictionary containing the centers of
        four ellipses in a specific order. The keys in the dictionary represent the positions of the
        ellipses ('top-left', 'top-right', 'bottom-right', 'bottom-left'), and the values are tuples
        representing the center coordinates of each
        :return: The `line_fitting` function returns a dictionary `linefit_points` containing the fitted
        points for each corner ('top-left', 'top-right', 'bottom-right', 'bottom-left') based on the
        ellipse centers and the corners of the region of interest (ROI). Additionally, it returns the
        `roi_corner_inorder` dictionary which contains the corners of the ROI in a specific order.
        """
    def line_fitting(self, roi_corners, ellipse_centers_inorder):
        def objective_function(t):
            D = tuple((A_i + t * (B_i - A_i) for A_i, B_i in zip(A, B)))
            dist_squared = sum((D_i - C_i)**2 for D_i, C_i in zip(D, C))
            return dist_squared
            
        roi_corner_tuple = [(int(x), int(y)) for x, y in roi_corners[0][0]]
        roi_corner_inorder=self.getCornerinOrder(roi_corner_tuple)
        linefit_points = {  
        'top-left': None,  
        'top-right': None,  
        'bottom-right': None,  
        'bottom-left': None 
        } 
        for keys, value in roi_corner_inorder.items():
            C=value
            if keys=='top-left' or keys=='bottom-right':
                A=ellipse_centers_inorder['top-left']
                B=ellipse_centers_inorder['bottom-right']
            elif keys=='top-right' or keys=='bottom-left':
                A=ellipse_centers_inorder['top-right']
                B=ellipse_centers_inorder['bottom-left']
            # 初始猜测参数t
            initial_guess = 0.5
            result = minimize(objective_function, initial_guess, method='BFGS')
            t_optimal = result.x[0]
            linefit_points[keys] = tuple((A_i + t_optimal * (B_i - A_i) for A_i, B_i in zip(A, B)))
 
        return linefit_points,roi_corner_inorder

####################SUB-PIXEL######################
    def mask_inner_marker(self,roi,roi_corners,roi_corner_inorder,subpix_path):
        """
        This function takes an image region of interest (ROI) and fills it with a color obtained from the
        center of the ROI.
        
        :param roi: The `roi` parameter in the `mask_inner_marker` function is typically an image or a
        region of interest (ROI) that you want to process. It is used for extracting color information and
        performing operations like filling polygons with a specific color
        :param roi_corners: The `roi_corners` parameter in the `mask_inner_marker` function seems to be a
        list containing the corners of a region of interest (ROI). The function converts the first element
        of `roi_corners` to a numpy array and then extracts the corner points as integers to be used for
        further processing
        :param roi_corner_inorder: The `roi_corner_inorder` parameter seems to be a dictionary containing
        the coordinates of the corners of a region of interest (ROI). The keys in this dictionary are likely
        'top-left', 'top-right', 'bottom-left', and 'bottom-right', each corresponding to a corner of the
        ROI
        :param subpix_path: The `subpix_path` parameter in the `mask_inner_marker` function is the file path
        where the processed image will be saved after filling the polygon with a specific color. This path
        is used in the `cv2.imwrite` function to save the image before displaying it
        :return: The function `mask_inner_marker` is returning the modified image `roi_subpix` after filling
        a polygon with a specific color at the center of the region of interest (ROI).
        """
        roi_subpix=roi
        roi_corners1=np.array(roi_corners[0]) 
        roi_corners_array = roi_corners1.astype(int)
        subpix_points = self.ExpandArea(K=0.8,quadrilateral_points=roi_corners_array[0])
        subpix_points_reshape_float = subpix_points[0].reshape(1,4,2)
        subpix_points_reshape_int = subpix_points_reshape_float.astype(int)
        pts = subpix_points_reshape_int.reshape((-1, 1, 2)) 
        x_center = (roi_corner_inorder['top-left'][0] + roi_corner_inorder['top-right'][0]) // 2  
        y_center = (roi_corner_inorder['top-left'][1] + roi_corner_inorder['top-right'][1]) // 2  +5
        color_at_center = roi[y_center, x_center]  
        assert len(color_at_center) == 3
        color_at_center = tuple(map(int, color_at_center))
        b,g,r = tuple(color_at_center)
        cv2.imwrite(subpix_path,roi_subpix)
        cv2.fillPoly(roi_subpix, [pts], (b,g,r)) 
        cv2.namedWindow('Filled Marker',cv2.WINDOW_NORMAL)  
        cv2.imshow('Filled Marker', roi_subpix)  
        cv2.waitKey(0)  
        cv2.destroyAllWindows() 
        return roi_subpix

    def euclidean_distance(self,point1, point2):
        """
        The function calculates the Euclidean distance between two points in a multi-dimensional space.
        
        :param point1: The `point1` parameter represents the coordinates of the first point in a
        multi-dimensional space. It could be a list, tuple, or array containing the coordinates of the
        first point
        :param point2: The `euclidean_distance` function you provided calculates the Euclidean distance
        between two points in n-dimensional space. The `point1` and `point2` parameters represent the
        coordinates of the two points in the form of arrays or lists. The function subtracts the
        coordinates of `point2` from
        :return: The function `euclidean_distance` calculates the Euclidean distance between two points
        `point1` and `point2` in n-dimensional space. The function returns the Euclidean distance as a
        numerical value.
        """
        return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))


    def find_subpixel_corners(self,roi_corners,subpix_corners):
        """
        The function `find_subpixel_corners` takes in two sets of corners and calculates the subpixel points
        closest to each corner of a region of interest.
        
        :param roi_corners: The `roi_corners` parameter seems to represent the corners of a region of
        interest (ROI), while the `subpix_corners` parameter likely represents the subpixel corners that are
        being compared to the ROI corners
        :param subpix_corners: The `subpix_corners` parameter in the `find_subpixel_corners` function
        represents a list of subpixel corners that are being compared to the corners of a region of interest
        (ROI) to find the closest subpixel corner to each ROI corner. These subpixel corners are likely
        represented as (x
        :return: The function `find_subpixel_corners` returns a dictionary `subpixel_points` containing the
        subpixel coordinates of the corners of a region of interest (ROI). The keys in the dictionary
        represent the corners ('top-right', 'bottom-right', 'bottom-left', 'top-left') and the values are
        the corresponding subpixel coordinates.
        """
        subpixel_points = {  
        'top-right': None,
        'bottom-right': None,
        'bottom-left': None,
        'top-left': None } 
        roi_corners1=np.array(roi_corners[0]) 
        roi_corners_array = roi_corners1.astype(int)
        corner_distance= [500,500,500,500]
        for i in range(len(roi_corners_array[0])):
            for j in range(len(subpix_corners)):
                if self.euclidean_distance(roi_corners_array[0][i], subpix_corners[j]) < corner_distance[i]:
                    corner_distance[i] = self.euclidean_distance(roi_corners_array[0][i], subpix_corners[j])
                    if i==0:
                        subpixel_points['top-right'] = subpix_corners[j]
                    if i==1:
                        subpixel_points['bottom-right'] = subpix_corners[j]
                    if i==2:
                        subpixel_points['bottom-left'] = subpix_corners[j]
                    if i==3:
                        subpixel_points['top-left'] = subpix_corners[j]

        return subpixel_points


    """
    The function calculates the final optimal points by averaging corresponding points from two input
    dictionaries.
    
    :param linefit_points: The `linefit_points` parameter seems to contain the coordinates of points
    obtained from a line fitting process. It likely includes the following keys: 'top-left',
    'top-right', 'bottom-left', and 'bottom-right', each pointing to a tuple representing a point
    :param subpixel_points: It seems like you were about to provide more information about the
    `subpixel_points` parameter but it got cut off. Could you please provide the details or let me know
    how I can assist you further with the `final_points` function?
    :return: The function `final_points` returns a dictionary `final_optimal_points` containing the
    average of corresponding values from the `linefit_points` and `subpixel_points` dictionaries for the
    keys 'top-left', 'top-right', 'bottom-left', and 'bottom-right'.
    """
    def final_points(self, linefit_points,subpixel_points):
        final_optimal_points = {  
        'top-right': None,
        'bottom-right': None,
        'bottom-left': None,
        'top-left': None }
        final_optimal_points['top-left']= (subpixel_points['top-left'][0]+linefit_points['top-left'])/2
        final_optimal_points['top-right']= (subpixel_points['top-right'][0]+linefit_points['top-right'])/2
        final_optimal_points['bottom-left']= (subpixel_points['bottom-left'][0]+linefit_points['bottom-left'])/2
        final_optimal_points['bottom-right']= (subpixel_points['bottom-right'][0]+linefit_points['bottom-right'])/2
        return final_optimal_points


if __name__=="__main__":
    image_path = "C:/02_FAU/S3/Project_Thesis/PIC/0424/3.png"
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
    image1=Image(id=1,image_path=image_path,aruco_dict=aruco_dict)
    corners, ids, _,image= image1.detect_markers()
    marker_area=image1.calculate_marker_area(corners)
    quadrilateral_points=image1.Get_quadrilaternal(corners)
    expand_points,w,h=image1.ExpandArea(K=3,quadrilateral_points=quadrilateral_points)
    roi=image1.Get_ROI(expand_points,image)
    ellipse_center,roi_corners =image1.Ellipse_detection(roi,w,h)
    ellipse_centers_inorder = image1.getCornerinOrder(ellipse_center)
    line_fitting_points,roi_corner_inorder =image1.line_fitting(roi_corners,ellipse_centers_inorder)
    subpix_path = subpix_path = "C:/02_FAU/S3/Project_Thesis/Code/Improved_Aruco_Marker/subpix_roi.jpg"
    roi_masked = image1.mask_inner_marker(roi,roi_corners,roi_corner_inorder,subpix_path)
    script_path = r"C:\02_FAU\S3\Project_Thesis\Code\Improved_Aruco_Marker\新建文件夹\Subpix.py"
    subprocess.run(["python", script_path])
    subpix_corners = get_subpix_corners()
    subpix_points = image1.find_subpixel_corners(roi_corners,subpix_corners)
    final_optimal_dict=image1.final_points(line_fitting_points,subpix_points)
    # Calculate the reporjection error
    pkl_path= "C:/02_FAU/S3/Project_Thesis/Code/Improved_Aruco_Marker/新建文件夹/cameraMatrix.pkl"
    dist_path= "C:/02_FAU/S3/Project_Thesis/Code/Improved_Aruco_Marker/新建文件夹/dist.pkl"
    cameraMatrix, distCoeffs = rp.get_camera_para(pkl_path,dist_path)
    final_corner_array = rp.ChangeToProperFormat_toReprojection(final_optimal_dict)
    marker_size = 0.05
    after_projection,before_projection,rotation_vector,translation_vector=rp.ReprojectionPoints(marker_size,final_corner_array, cameraMatrix, distCoeffs)
    reprojection_error = rp.ReprojectionError(after_projection=after_projection,before_projection=before_projection)
    print(f'The reprojection error of image {os.path.basename(image_path)} is: \n {reprojection_error[0]}')



