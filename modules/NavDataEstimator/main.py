"""
TBD
"""
import argparse
import concurrent.futures
import cv2
import numpy as np
import os
import pickle
import sys

from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.NavDataEstimator.src.navdataestimator import NavDataEstimator
from modules.NavDataEstimator.src.utils.enums import RectificationMode
from modules.NavDataEstimator.src.utils.helpers import crop_roi, draw_distance, draw_depth_map, \
                                                       draw_horizontal_lines


__author__ = "EnriqueMoran"


class MainApp:
    """
    This class executes the main loop of NavDataEstimator module.

    Args:
        - args (argparse.Namespace): Arguments to update loggers (Main App) parameters. 
        - log_filepath (str): Default path to store Main App messages log file.
        - log_format (str): Main App default logger format.
        - log_level (str): Main App default logger level.
    """

    def __init__(self, log_format:str, log_level:str, log_filepath:str, 
                 config_filepath:str, args:argparse.Namespace):
        self.log_filepath   = log_filepath
        self.log_format     = log_format
        self.log_level      = log_level
        self.logger         = None

        self.config_filepath = config_filepath

        log_dir = os.path.dirname(self.log_filepath)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self._check_args(args)


    def _check_args(self, args) -> None:
        """
        Check passed args.

        Args:
            - args (argparse.Namespace): Passed args to be checked.
        """
        valid_log_levels = ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]
        if args.level:
            if args.level.upper() in valid_log_levels:
                self.log_level = os.environ.get("LOGLEVEL", args.level.upper())
            else:
                message = f"Warning: logging level {args.level} not found within valid values: " +\
                          f"{valid_log_levels}. Defaulting to {self.log_level}."
                print(message)

        if args.log:
            log_path = Path(args.log)
            if log_path.exists():
                self.log_filepath = args.log
            else:
                message = f"Warning: logging file path {log_path} not found. " +\
                          f"Defaulting to {self.log_filepath}."
                print(message)
        
        if not args.keep_logs:
            if os.path.exists(self.log_filepath):
                with open(self.log_filepath, 'w'):
                    pass


    def run(self):
        self.test3()
    

    def test3(self):
        nav_data_estimator = NavDataEstimator(filename=self.log_filepath, 
                                              format=self.log_format, 
                                              level=self.log_level, 
                                              config_path=self.config_filepath)


        display_size = (750, 600)

        #params = nav_data_estimator.distance_calculator.calibrator.calibrate_cameras()
        #if params:
        #    err, Kl, Dl, Kr, Dr, R, T, E, F, pattern_points, left_pts, right_pts = params
        #else:
        #    print(f"Error calibrating cameras. Aborting!")
        #    return

        params = nav_data_estimator.distance_calculator.calibrator.load_calibration()
        
        err, Kl, Dl, Kr, Dr, R, T, E, F, pattern_points, left_pts, right_pts = \
        params["err"], params["Kl"], params["Dl"], params["Kr"], params["Dr"], params["R"], \
        params["T"], params["E"], params["F"], params["pattern_points"], params["left_pts"], \
        params["right_pts"]

        image_left  = cv2.imread("./modules/NavDataEstimator/test/video_2_2_test_left.png")
        image_right = cv2.imread("./modules/NavDataEstimator/test/video_2_2_test_right.png")

        rect_left, rect_right, params = nav_data_estimator.distance_calculator.rectify_images(
            image_left=image_left, image_right=image_right, Kl=Kl, Dl=Dl, Kr=Kr, Dr=Dr, R=R, T=T
        )

        h_lines_left  = draw_horizontal_lines(cv2.resize(rect_left, display_size), 
                                              line_interval=20)
        h_lines_right = draw_horizontal_lines(cv2.resize(rect_right, display_size), 
                                              line_interval=20)

        combined_image = cv2.hconcat([cv2.resize(h_lines_left, display_size),
                                      cv2.resize(h_lines_right, display_size)])
        cv2.imshow('Rectified images', combined_image)
        cv2.waitKey(0)
        
        n_disp     = nav_data_estimator.distance_calculator.config_parser.parameters.num_disparities
        block_size = nav_data_estimator.distance_calculator.config_parser.parameters.block_size
        max_disp   = 160

        stereo_bm = cv2.StereoBM_create(n_disp, block_size)
        dispmap_bm = stereo_bm.compute(rect_left, rect_right)

        stereo_sgbm = cv2.StereoSGBM_create(0, max_disp, block_size)
        dispmap_sgbm = stereo_sgbm.compute(rect_left, rect_right)

        dispmap_bm = nav_data_estimator.distance_calculator.normalize_depth_map(dispmap_bm)

        dispmap_sgbm = nav_data_estimator.distance_calculator.normalize_depth_map(dispmap_sgbm)

        calibration_mode = nav_data_estimator.distance_calculator.rectification_mode

        if calibration_mode == RectificationMode.CALIBRATED_SYSTEM:
            params['xmap'] = params['xmap_l']
            params['ymap'] = params['ymap_l']
        elif calibration_mode == RectificationMode.UNCALIBRATED_SYSTEM:
            params['H'] = params['Hl']

        undistorded_bm = nav_data_estimator.distance_calculator.undistort_rectified_image(
            dispmap_bm, **params
        )

        undistorded_sgbm = nav_data_estimator.distance_calculator.undistort_rectified_image(
            dispmap_sgbm, **params
        )

        image_left  = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
        image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

        draw_depth_bm   = draw_depth_map(image_left, undistorded_bm)
        draw_depth_sgbm = draw_depth_map(image_left, undistorded_sgbm)
        
        combined_image = cv2.hconcat([cv2.resize(draw_depth_bm, display_size), cv2.resize(draw_depth_sgbm, display_size)])
        cv2.imshow('Depth maps', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def test2(self):
        nav_data_estimator = NavDataEstimator(filename=self.log_filepath, 
                                              format=self.log_format, 
                                              level=self.log_level, 
                                              config_path=self.config_filepath)

        video_left  = Path("./modules/NavDataEstimator/calibration/videos/20240822/calibration_1_left.mp4")
        video_right = Path("./modules/NavDataEstimator/calibration/videos/20240822/calibration_1_right.mp4")

        # Error, left camera matrix, left camera distortion, right camera matrix, right camera distortion
        # Rotation matrix, Translation matrix, Essential matrix, Fundamental matrix
       # err, Kl, Dl, Kr, Dr, R, T, E, F, pattern_points, left_pts, right_pts = \
       #     nav_data_estimator.distance_calculator.calibrate_cameras_video(video_left, video_right)
       # print(f"Reprojection error: {err}")
       # nav_data_estimator.distance_calculator.left_calibrator.save_calibration(err, Kl, Dl, Kr, Dr,
       #                                                                          R, T, E, F,
       #                                                                          pattern_points,
       #                                                                          left_pts,
       #                                                                          right_pts)

        params = nav_data_estimator.distance_calculator.left_calibrator.load_calibration()
        
        err, Kl, Dl, Kr, Dr, R, T, E, F, pattern_points, left_pts, right_pts = \
        params["err"], params["Kl"], params["Dl"], params["Kr"], params["Dr"], params["R"], \
        params["T"], params["E"], params["F"], params["pattern_points"], params["left_pts"], \
        params["right_pts"]

        display_size = (800, 600)

        img_l = cv2.imread("./modules/NavDataEstimator/test/video_1_1_test_left.png",  cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread("./modules/NavDataEstimator/test/video_1_1_test_right.png", cv2.IMREAD_GRAYSCALE)

        img_size = img_l.shape[:2][::-1]
        R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(Kl, Dl, Kr, Dr, img_size, R, T)
        xmap1, ymap1 = cv2.initUndistortRectifyMap(Kl, Dl, R1, P1, img_size, cv2.CV_32FC1)
        xmap2, ymap2 = cv2.initUndistortRectifyMap(Kr, Dr, R2, P2, img_size, cv2.CV_32FC1)

        left_img_rectified  = cv2.remap(img_l, xmap1, ymap1, cv2.INTER_LINEAR)
        right_img_rectified = cv2.remap(img_r, xmap2, ymap2, cv2.INTER_LINEAR)

        #left_img_rectified  = cv2.remap(img_l, xmap1, ymap1, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        #right_img_rectified = cv2.remap(img_r, xmap2, ymap2, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        left_roi = crop_roi(left_img_rectified, validRoi1)
        right_roi = crop_roi(right_img_rectified, validRoi2)

        combined_image = cv2.hconcat([cv2.resize(left_roi, display_size), cv2.resize(right_roi, display_size)])
        cv2.imshow('Rectified images', combined_image)
        cv2.waitKey(0)

        #undistorted_l = cv2.resize(undistorted_l, display_size)
        #undistorted_r = cv2.resize(undistorted_r, display_size)

        #depth_map  = nav_data_estimator.distance_calculator.get_depth_map(left_image=left_img_rectified,
        #                                                                  right_image=right_img_rectified,
        #                                                                  n_disparities=0,
        #                                                                  block_size=15)
        #normalized_depth_map = nav_data_estimator.distance_calculator.normalize_depth_map(depth_map)

        #undistorted_l = cv2.resize(undistorted_l, display_size)
        #normalized_depth_map     = cv2.resize(normalized_depth_map, display_size)
        #draw_map = draw_depth_map(undistorted_l, normalized_depth_map)

        n_disp = 0
        block_size = 27
        max_disp = 128

        stereo_bm = cv2.StereoBM_create(n_disp, block_size)
        dispmap_bm = stereo_bm.compute(left_img_rectified, right_img_rectified)

        stereo_sgbm = cv2.StereoSGBM_create(0, max_disp, block_size)
        dispmap_sgbm = stereo_sgbm.compute(left_img_rectified, right_img_rectified)

        dispmap_bm = nav_data_estimator.distance_calculator.normalize_depth_map(dispmap_bm)

        dispmap_sgbm = nav_data_estimator.distance_calculator.normalize_depth_map(dispmap_sgbm)

        cv2.imshow('dispmap_bm', cv2.resize(dispmap_bm, display_size))
        cv2.waitKey(0)

        cv2.imshow('dispmap_sgbm', cv2.resize(dispmap_sgbm, display_size))
        cv2.waitKey(0)

        #dispmap_bm = crop_roi(dispmap_bm, validRoi1)
        #dispmap_sgbm = crop_roi(dispmap_sgbm, validRoi1)
        #img_l = crop_roi(img_l, validRoi1)

        #angle = -3
        #center = (dispmap_bm.shape[1] // 2, dispmap_bm.shape[0] // 2)
        #rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        #rotated_dispmap = cv2.warpAffine(dispmap_bm, rotation_matrix, (dispmap_bm.shape[1], dispmap_bm.shape[0]))

        #center = (dispmap_sgbm.shape[1] // 2, dispmap_sgbm.shape[0] // 2)
        #rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        #rotated_dispmap2 = cv2.warpAffine(dispmap_sgbm, rotation_matrix, (dispmap_sgbm.shape[1], dispmap_sgbm.shape[0]))

        #dx, dy = 40, 40 
        #translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        #aligned_dispmap = cv2.warpAffine(rotated_dispmap, translation_matrix, (rotated_dispmap.shape[1], rotated_dispmap.shape[0]))
        #aligned_dispmap2 = cv2.warpAffine(rotated_dispmap2, translation_matrix, (rotated_dispmap2.shape[1], rotated_dispmap2.shape[0]))

        draw_dist_1 = draw_depth_map(img_l, dispmap_bm)
        draw_dist_2 = draw_depth_map(img_l, dispmap_sgbm)

        focal_len = nav_data_estimator.config_parser.left_camera_specs.focal_length
        pixel_size = nav_data_estimator.config_parser.left_camera_specs.pixel_size
        baseline = nav_data_estimator.config_parser.system_setup.baseline_distance

        distance_map_1 = nav_data_estimator.distance_calculator.get_distance_map(dispmap_bm, focal_len,
                                                                                 pixel_size, baseline)
        
        distance_map_2 = nav_data_estimator.distance_calculator.get_distance_map(dispmap_sgbm, focal_len,
                                                                                 pixel_size, baseline)
                                                                                 

        combined_image = cv2.hconcat([cv2.resize(draw_dist_1, display_size), cv2.resize(draw_dist_2, display_size)])
        cv2.imshow('Res', combined_image)
        cv2.waitKey(0)


        margin = 50
        step = 100
        #image_width, image_height = nav_data_estimator.config_parser.parameters.resolution
        image_width, image_height = (800, 600)
        x_points = np.arange(margin, image_width - margin, step)
        y_points = np.arange(margin, image_height - margin, step)

        points = [(int(x), int(y)) for x in x_points for y in y_points]

        dist1 = draw_distance(img_l, distance_map_1, points)
        cv2.imshow('dist 1', cv2.resize(dist1, display_size))
        cv2.waitKey(0)

        dist2 = draw_distance(img_l, distance_map_2, points)
        cv2.imshow('dist 2', cv2.resize(dist2, display_size))
        cv2.waitKey(0)
        
        
        cv2.destroyAllWindows()



    def test(self):
        nav_data_estimator = NavDataEstimator(filename=self.log_filepath, 
                                              format=self.log_format, 
                                              level=self.log_level, 
                                              config_path=self.config_filepath)
        
        #nav_data_estimator.distance_calculator.calibrate_cameras_video(video_path_l=video_left,
        #                                                               video_path_r=video_right)
        
        image_size = nav_data_estimator.config_parser.parameters.resolution
        
        img_l = cv2.imread("./modules/NavDataEstimator/test/video_1_1_test_left.png",  cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread("./modules/NavDataEstimator/test/video_1_1_test_right.png", cv2.IMREAD_GRAYSCALE)

        with open("./modules/NavDataEstimator/calibration/params/calibration_left.pkl", 'rb') as file:
            camera_matrix_l, dist_l, _, _, obj_points_list_l, img_points_list_l = pickle.load(file)
        
        with open("./modules/NavDataEstimator/calibration/params/calibration_right.pkl", 'rb') as file:
            camera_matrix_r, dist_r, _, _, _, img_points_list_r = pickle.load(file)

        camera_matrix_l, roi_l = cv2.getOptimalNewCameraMatrix(
            camera_matrix_l, dist_l, image_size, 
            alpha=nav_data_estimator.config_parser.parameters.alpha, centerPrincipalPoint=True
        )
        camera_matrix_r, roi_r = cv2.getOptimalNewCameraMatrix(
            camera_matrix_r, dist_r, image_size,
            alpha=nav_data_estimator.config_parser.parameters.alpha, centerPrincipalPoint=True
        )

        rectified_images = nav_data_estimator.distance_calculator.get_rectified_images(
            image_left=img_l,
            image_right=img_r,
            obj_points_list_l=obj_points_list_l,
            img_points_list_l=img_points_list_l,
            img_points_list_r=img_points_list_r,
            camera_matrix_l=camera_matrix_l,
            dist_l=dist_l,
            camera_matrix_r=camera_matrix_r,
            dist_r=dist_r
        )

        display_size = (800, 600)
        rectified_left, rectified_right, roi_l, roi_r = rectified_images

        #rectified_left_roi  = cv2.resize(draw_roi(rectified_left, roi_l), (800, 600))
        #rectified_right_roi = cv2.resize(draw_roi(rectified_right, roi_l), (800, 600))

        #combined_image = cv2.hconcat([cv2.resize(img_l, display_size), cv2.resize(img_r, display_size)])
        #cv2.imshow('Original images', combined_image)
        #cv2.waitKey(0)


        sift = cv2.SIFT_create()
        keypoints_left, descriptors_left   = sift.detectAndCompute(img_l, None)
        keypoints_right, descriptors_right = sift.detectAndCompute(img_r, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        pts1 = [keypoints_left[m.queryIdx].pt for m in good_matches]
        pts2 = [keypoints_right[m.trainIdx].pt for m in good_matches]

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        # We select only inlier points
        pts1 = pts1[inliers.ravel() == 1]
        pts2 = pts2[inliers.ravel() == 1]
        
        def drawlines(img1src, img2src, lines, pts1src, pts2src):
            ''' img1 - image on which we draw the epilines for the points in img2
                lines - corresponding epilines '''
            r, c = img1src.shape
            img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
            img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
            # Edit: use the same random seed so that two images are comparable!
            np.random.seed(0)
            for r, pt1, pt2 in zip(lines, pts1src, pts2src):
                color = tuple(np.random.randint(0, 255, 3).tolist())
                x0, y0 = map(int, [0, -r[2]/r[1]])
                x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
                img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
                img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
                img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
            return img1color, img2color
        
        lines1 = cv2.computeCorrespondEpilines(
            pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = drawlines(img_l, img_r, lines1, pts1, pts2)

        lines2 = cv2.computeCorrespondEpilines(
            pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = drawlines(img_r, img_l, lines2, pts2, pts1)

        h1, w1 = img_l.shape
        h2, w2 = img_r.shape
        _, H1, H2 = cv2.stereoRectifyUncalibrated(
            np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
        )

        img1_rectified = cv2.warpPerspective(img_l, H1, (w1, h1))
        img2_rectified = cv2.warpPerspective(img_r, H2, (w2, h2))

        block_size = 7
        min_disp = -128
        max_disp = 128

        num_disp = max_disp - min_disp
        uniquenessRatio = 5
        speckleWindowSize = 200
        speckleRange = 2
        disp12MaxDiff = 0

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            disp12MaxDiff=disp12MaxDiff,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )
        disparity_SGBM = stereo.compute(img1_rectified, img2_rectified)

        disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                              beta=0, norm_type=cv2.NORM_MINMAX)
        disparity_SGBM = np.uint8(disparity_SGBM)
        cv2.imshow("Disparity", disparity_SGBM)
        cv2.waitKey(0)

        #combined_image = cv2.hconcat([img5, img3])
        #cv2.imshow('Rectified with lines', combined_image)
        #cv2.waitKey(0)

        #rectified_left_with_lines = draw_horizontal_lines(image=rectified_left_roi,
        #                                                  line_interval=50,
        #                                                  color=(0, 255, 0),
        #                                                  thickness=2)

        #rectified_right_with_lines = draw_horizontal_lines(image=rectified_right_roi,
        #                                                   line_interval=50,
        #                                                   color=(0, 255, 0), 
        #                                                   thickness=2)
        


        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MADTS Video Synchronizer.")

    parser.add_argument("--level", 
                        type=str,
                        help="Set loggin level. Valid values: DEBUG, INFO, WARN, ERROR, CRITICAL.")

    parser.add_argument("--log", 
                        type=str,
                        help="Set logging file path.")

    parser.add_argument("--keep_logs",
                        type=bool,
                        help="If enabled, won't clear logs files on new run.")

    args = parser.parse_args([])

    log_filepath   = f"./modules/NavDataEstimator/logs/{datetime.now().strftime('%Y%m%d')}.log"
    log_format     = '%(asctime)s - %(levelname)s - %(name)s::%(funcName)s - %(message)s'
    log_level      = os.environ.get("LOGLEVEL", "DEBUG")

    config_filepath = f"./modules/NavDataEstimator/cfg/config.ini"
    
    app = MainApp(log_filepath=log_filepath, log_format=log_format,
                  log_level=log_level, config_filepath=config_filepath, args=args)

    app.run()