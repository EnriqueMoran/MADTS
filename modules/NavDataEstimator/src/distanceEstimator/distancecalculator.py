"""
Implements calibration managemer class.
"""

import cv2
import numpy as np

from modules.NavDataEstimator.src.baseclass import BaseClass
from modules.NavDataEstimator.src.configmanager import ConfigManager
from modules.NavDataEstimator.src.distanceEstimator.cameracalibrator import Calibrator
from modules.NavDataEstimator.src.utils.enums import RectificationMode, UndistortMethod
from modules.NavDataEstimator.src.utils.helpers import crop_roi, draw_distance, \
                                                       draw_epipolar_lines, draw_horizontal_lines


__author__ = "EnriqueMoran"


class DistanceCalculator(BaseClass):
    """
    TBD
    
    Args:
        - config_path (str): Path to configuration file.
        - filename (str): Path to store log file; belongs to BaseClass.
        - format (str): Logger format; belongs to BaseClass.
        - level (str): Logger level; belongs to BaseClass.
    """
    def __init__(self, filename:str, format:str, level:str, config_path:str):
        super().__init__(filename, format, level)
        self.config_parser = ConfigManager(filename=filename, format=format, level=level, 
                                           config_path=config_path)
        self.calibrator = Calibrator(filename=filename, format=format, level=level,
                                     config_path=config_path)
        self.undistort_method   = self.config_parser.parameters.undistort_method
        self.rectification_mode = self.config_parser.parameters.rectification_mode
        self.detection_kernel   = self.config_parser.parameters.detection_kernel
        self.horizontal_fov = self.config_parser.left_camera_specs.h_fov    # Relative to left cam
        self.vertical_fov   = self.config_parser.left_camera_specs.v_fov    # Relative to left cam

    
    def precompute_rectification_maps(self, Kl, Dl, Kr, Dr, image_size, R, T):
        """
        TBD
        """
        Rl, Rr, Pl, Pr, Q, roi_l, roi_r = cv2.stereoRectify(Kl, Dl, Kr, Dr, image_size, R, T)
        xmap_l, ymap_l = cv2.initUndistortRectifyMap(Kl, Dl, Rl, Pl, image_size, cv2.CV_32FC1)
        xmap_r, ymap_r = cv2.initUndistortRectifyMap(Kr, Dr, Rr, Pr, image_size, cv2.CV_32FC1)
        res = {
                "xmap_l": xmap_l,
                "ymap_l": ymap_l,
                "xmap_r": xmap_r,
                "ymap_r": ymap_r,
                "roi_l":  roi_l,
                "roi_r":  roi_r,
                "Q": Q
              }
        return res

    
    def rectify_images(self, image_left, image_right, **kwargs):
        """
        TBD
        Args:
        image_left (numpy.ndarray): The left image to rectify.
        image_right (numpy.ndarray): The right image to rectify.
        **kwargs: Contains calibration parameters:
            - Kl: Intrinsic matrix of the left camera.
            - Dl: Distortion coefficients of the left camera.
            - Kr: Intrinsic matrix of the right camera.
            - Dr: Distortion coefficients of the right camera.
            - img_size: Size of the image (width, height).
            - R: Rotation matrix between the coordinate systems of the cameras.
            - T: Translation vector between the coordinate systems of the cameras.

        Returns:
            rectified_left (numpy.ndarray): The rectified left image.
            rectified_right (numpy.ndarray): The rectified right image.
        """
        rectified_left  = None
        rectified_right = None
        params = dict()    # Contains params according to used rectification mode

        image_left  = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
        image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

        if self.rectification_mode == RectificationMode.CALIBRATED_SYSTEM:
            Kl = kwargs.get('Kl')
            Dl = kwargs.get('Dl')
            Kr = kwargs.get('Kr')
            Dr = kwargs.get('Dr')
            R = kwargs.get('R')
            T = kwargs.get('T')

            image_size = image_left.shape[:2][::-1]
            Rl, Rr, Pl, Pr, Q, roi_l, roi_r = cv2.stereoRectify(Kl, Dl, Kr, Dr, image_size, R, T)

            xmap_l, ymap_l = cv2.initUndistortRectifyMap(Kl, Dl, Rl, Pl, image_size, cv2.CV_32FC1)
            xmap_r, ymap_r = cv2.initUndistortRectifyMap(Kr, Dr, Rr, Pr, image_size, cv2.CV_32FC1)

            rectified_left  = cv2.remap(image_left, xmap_l, ymap_l, cv2.INTER_LINEAR)
            rectified_right = cv2.remap(image_right, xmap_r, ymap_r, cv2.INTER_LINEAR)

            params['xmap_l'] = xmap_l
            params['ymap_l'] = ymap_l
            params['xmap_r'] = xmap_r
            params['ymap_r'] = ymap_r
            params['roi_l']  = roi_l
            params['roi_r']  = roi_r

        elif self.rectification_mode == RectificationMode.UNCALIBRATED_SYSTEM:
            # https://stackoverflow.com/questions/36172913/opencv-depth-map-from-uncalibrated-stereo-system
            # https://ashwinvadivel.com/2021/10/stereo-image-rectification/
            # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html         TODO CHECK
            # https://www.andreasjakl.com/how-to-apply-stereo-matching-to-generate-depth-maps-part-3/
            # https://docs.opencv.org/4.2.0/d3/d14/tutorial_ximgproc_disparity_filtering.html
            # https://learnopencv.com/homography-examples-using-opencv-python-c/
            
            sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.04, edgeThreshold=10)
            keypoints_left, descriptors_left   = sift.detectAndCompute(image_left, None)
            keypoints_right, descriptors_right = sift.detectAndCompute(image_right, None)

            FLANN_INDEX_KDTREE = 0
            index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            pts_l = [keypoints_left[m.queryIdx].pt for m in good_matches]
            pts_r = [keypoints_right[m.trainIdx].pt for m in good_matches]

            pts_l = np.int32(pts_l)
            pts_r = np.int32(pts_r)
            fundamental_matrix, inliers = cv2.findFundamentalMat(pts_l, pts_r, cv2.FM_RANSAC)

            pts_l = pts_l[inliers.ravel() == 1]
            pts_r = pts_r[inliers.ravel() == 1]

            width, height = image_left.shape[1], image_left.shape[0]

            _, Hl, Hr = cv2.stereoRectifyUncalibrated(
                np.float32(pts_l), np.float32(pts_r), fundamental_matrix, imgSize=(width, height)
            )

            # image_left.shape should be the same as image_right.shape
            rectified_left  = cv2.warpPerspective(image_left, Hl, (width, height))
            rectified_right = cv2.warpPerspective(image_right, Hr, (width, height))

            params['Hl'] = Hl
            params['Hr'] = Hr

            rectified_corners_left = cv2.perspectiveTransform(np.array([[[0, 0], [0, height], [width, 0], [width, height]]], dtype=np.float32), Hl)[0]
            rectified_corners_right = cv2.perspectiveTransform(np.array([[[0, 0], [0, height], [width, 0], [width, height]]], dtype=np.float32), Hr)[0]

            x_min_left, y_min_left = np.int32(rectified_corners_left.min(axis=0))
            x_max_left, y_max_left = np.int32(rectified_corners_left.max(axis=0))

            x_min_right, y_min_right = np.int32(rectified_corners_right.min(axis=0))
            x_max_right, y_max_right = np.int32(rectified_corners_right.max(axis=0))

            roi_l = (x_min_left, y_min_left, x_max_left - x_min_left, y_max_left - y_min_left)
            roi_r = (x_min_right, y_min_right, x_max_right - x_min_right, y_max_right - y_min_right)

            params['roi_l'] = roi_l
            params['roi_r'] = roi_r

            ######################################## DEBUG ########################################
            #lines_left_from_right = cv2.computeCorrespondEpilines(pts_r.reshape(-1, 1, 2), 2, 
            #                                                      fundamental_matrix)
            #lines_left_from_right = lines_left_from_right.reshape(-1, 3)
            #image_left_epi_right, _ = draw_epipolar_lines(image_left, image_right, 
            #                                              lines_left_from_right, pts_l, pts_r)
            #
            #lines_right_from_left = cv2.computeCorrespondEpilines(pts_l.reshape(-1, 1, 2), 1, 
            #                                                      fundamental_matrix)
            #lines_right_from_left = lines_right_from_left.reshape(-1, 3)
            #image_right_epi_left, _ = draw_epipolar_lines(image_right, image_left, 
            #                                              lines_right_from_left, pts_r,pts_l)
            #
            #display_size = (750, 600)
            #combined_image = cv2.hconcat([cv2.resize(image_left_epi_right, display_size), 
            #                              cv2.resize(image_right_epi_left, display_size)])
            #cv2.imshow('Rectified images', combined_image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #######################################################################################

        else:
            error_msg = f"Rectification mode {self.rectification_mode} not recognized. Aborting!"
            self.logger.error(error_msg)
            return None
        
        return rectified_left, rectified_right, params
    

    def undistort_rectified_image(self, image, **kwargs):
        """
        TBD
        """
        undistorted_image = None
        if self.rectification_mode == RectificationMode.CALIBRATED_SYSTEM:
            xmap = kwargs.get('xmap')
            ymap = kwargs.get('ymap')

            xmap_inv = np.zeros_like(xmap)
            ymap_inv = np.zeros_like(ymap)

            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    x = int(xmap[i, j])
                    y = int(ymap[i, j])
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        xmap_inv[y, x] = j
                        ymap_inv[y, x] = i

            undistorted_image = cv2.remap(image, xmap_inv, ymap_inv, cv2.INTER_LINEAR)
            
        elif self.rectification_mode == RectificationMode.UNCALIBRATED_SYSTEM:
            H = kwargs.get('H')
            H_inv = np.linalg.inv(H)

            undistorted_image = cv2.warpPerspective(image, H_inv, (image.shape[1], image.shape[0]))

        else:
            error_msg = f"Rectification mode {self.rectification_mode} not recognized. Aborting!"
            self.logger.error(error_msg)
            return None
        
        return undistorted_image


    def get_depth_map(self, left_image:np.ndarray, right_image:np.ndarray, n_disparities:int=0, 
                      block_size:int=21):
        """
        TBD
        """
        height_l, width_l = left_image.shape[:2]
        height_r, width_r = right_image.shape[:2]

        avg_width  = (width_l + width_r) // 2
        avg_height = (height_l + height_r) // 2

        resized_img_l = cv2.resize(left_image, (avg_width, avg_height))
        resized_img_r = cv2.resize(right_image, (avg_width, avg_height))

        stereo = cv2.StereoBM_create(numDisparities=n_disparities, blockSize=block_size)
        return stereo.compute(resized_img_l, resized_img_r)


    def normalize_depth_map(self, depth_map):
        """
        TBD
        """
        image_copy = depth_map.copy()

        depth_map_normalized = cv2.normalize(image_copy, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_normalized = np.uint8(depth_map_normalized)
        depth_map_colored    = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_VIRIDIS)
        return depth_map_colored
    

    def get_distance_map(self, depth_map, focal_length, pixel_size, baseline):
        """
        TBD
        """
        focal_length_pixels = (focal_length / pixel_size) * 1000

        disparity_map = np.float32(depth_map)
        disparity_map[disparity_map == 0] = 1e-6

        distance_map = (focal_length_pixels * baseline) / disparity_map
        return distance_map


    def apply_disparity_filter(self, disp_map, stereo, image_left, image_right):
        """
        TBD
        """
        lmbda = 8000
        sigma = 1.5
        right_matcher = cv2.ximgproc.createRightMatcher(stereo)
        disparity_right = right_matcher.compute(image_right, image_left)

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        filtered_disp = wls_filter.filter(disp_map, image_left, 
                                          disparity_map_right=disparity_right)
        filtered_disp = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return filtered_disp


    def process_frame(self, frame_left, frame_right, nav_data_estimator, precomputed_maps, roi1, 
                      focal_length_l, pixel_size_l, baseline, points):
        frame_left_resized = cv2.resize(frame_left, 
                                        nav_data_estimator.config_parser.parameters.resolution)
        frame_right_resized = cv2.resize(frame_right, 
                                         nav_data_estimator.config_parser.parameters.resolution)

        frame_left_gray = cv2.cvtColor(frame_left_resized, cv2.COLOR_BGR2GRAY)
        frame_right_gray = cv2.cvtColor(frame_right_resized, cv2.COLOR_BGR2GRAY)

        # Rectify images using precomputed maps
        rectified_left = cv2.remap(frame_left_gray, precomputed_maps['map_left_x'], 
                                   precomputed_maps['map_left_y'], cv2.INTER_LINEAR)
        rectified_right = cv2.remap(frame_right_gray, precomputed_maps['map_right_x'], 
                                    precomputed_maps['map_right_y'], cv2.INTER_LINEAR)

        # Depth map calculation
        depth_map = nav_data_estimator.distance_calculator.get_depth_map(
            left_image=rectified_left,
            right_image=rectified_right,
            n_disparities=nav_data_estimator.config_parser.parameters.num_disparities,
            block_size=nav_data_estimator.config_parser.parameters.block_size
        )

        # Normalize and crop the depth map
        normalized_depth_map = nav_data_estimator.distance_calculator.normalize_depth_map(depth_map)
        normalized_depth_map = crop_roi(normalized_depth_map, roi1)

        # Compute distance map
        distance_map_left = nav_data_estimator.distance_calculator.get_distance_map(
            depth_map, focal_length_l, pixel_size_l, baseline
        )

        # Draw distances on the left frame
        frame_with_distances = draw_distance(frame_left_resized, distance_map_left, points)

        return frame_with_distances


    def get_homography(self, image1, image2):
        """
        TBD
        """
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(image1, None)
        kp2, des2 = orb.detectAndCompute(image2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H


    def get_avg_distance(self, depth_map, point):
        """
        Returns average distance from points within kernel area, excluding outliers.
        """
        avg_dist = 0
        kernel_half = self.detection_kernel // 2

        y, x = point
        x_min = max(x - kernel_half, 0)
        x_max = min(x + kernel_half, depth_map.shape[1] - 1)
        y_min = max(y - kernel_half, 0)
        y_max = min(y + kernel_half, depth_map.shape[0] - 1)

        kernel_area = depth_map[y_min:y_max + 1, x_min:x_max + 1]
        valid_values = kernel_area[kernel_area > 0]

        if valid_values.size > 0:
            mean_dist = valid_values.mean()
            std_dist = valid_values.std()

            threshold = 1 

            filtered_values = valid_values[
                (valid_values >= mean_dist - threshold * std_dist) & 
                (valid_values <= mean_dist + threshold * std_dist)
            ]

            if filtered_values.size > 0:
                avg_dist = float(filtered_values.mean())
            else:
                avg_dist = float('nan')
        else:
            avg_dist = float('nan')

        return avg_dist


    def get_angle(self, point, image_width, image_height):
        """
        Calculate the angle of a point with respect to the center of the image.
        
        Args:
            point (tuple): (x, y) coordinates of the point in the image.
            image_width (int): Width of the image.
            image_height (int): Height of the image.
        
        Returns:
            angle_x (float): Angle with respect to the x-axis (horizontal).
            angle_y (float): Angle with respect to the y-axis (vertical).
        """
        center_x = image_width / 2
        center_y = image_height / 2
        
        dx = point[0] - center_x
        dy = point[1] - center_y

        angle_per_pixel_x = self.horizontal_fov / image_width
        angle_per_pixel_y = self.vertical_fov / image_height
        
        angle_x = dx * angle_per_pixel_x
        angle_y = dy * angle_per_pixel_y
        
        return angle_x, angle_y