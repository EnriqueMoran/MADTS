from dataclasses import dataclass
from modules.NavDataEstimator.src.utils.enums import CalibrationMode, RectificationMode, \
                                                     UndistortMethod

__author__ = "EnriqueMoran"


@dataclass
class CameraSpecs:
    focal_length: float = None
    pixel_size:   float = None
    fov: float          = None


@dataclass
class SystemSetup:
    baseline_distance: float = None


@dataclass
class CalibrationSpecs:
    chessboard_width:  int            = None
    chessboard_height: int            = None
    chessboard_square_size: int       = None
    calibration_mode: CalibrationMode = None
    save_calibration_images: bool     = None
    save_calibration_params: bool     = None
    video_calibration_step:  int      = None
    save_calibration_images_path: str = None
    calibration_images_dir_left:  str = None
    calibration_images_dir_right: str = None
    calibration_video_left:  str      = None
    calibration_video_right: str      = None
    save_calibration_params_path: str = None
    load_calibration_params_path: str = None


@dataclass
class Parameters:
    num_disparities: int                  = None
    block_size: int                       = None
    gaussian_kernel_size: int             = None
    rectify_alpha: float                  = None
    undistort_method:   UndistortMethod   = None
    rectification_mode: RectificationMode = None


@dataclass
class Stream:
    left_camera:  str   = None
    right_camera: str   = None
    record_path:  str   = None
    max_detections: int = None
    record: bool     = None
    scale:  float    = None
    lost_frames: int = None
    

@dataclass
class Communication:
    group: str = None
    port:  int = None
    iface: str = None
    ttl:   int = None
    frequency: float = None


@dataclass
class Correlation:
    min_distance:float = None