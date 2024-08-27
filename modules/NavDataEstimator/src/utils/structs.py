from dataclasses import dataclass
from modules.NavDataEstimator.src.utils.enums import UndistortMethod

__author__ = "EnriqueMoran"


@dataclass
class CameraSpecs:
    focal_length: float = None
    pixel_size: float   = None
    fov: float          = None


@dataclass
class SystemSetup:
    baseline_distance: float = None


@dataclass
class CalibrationSpecs:
    image_directory: str              = None
    chessboard_width: int             = None
    chessboard_height: int            = None
    chessboard_square_size: int       = None
    frame_width: int                  = None
    frame_height: int                 = None
    save_calibration_images: bool     = None
    save_calibration_images_path: str = None
    save_calibration_params: bool     = None
    save_calibration_params_path: str = None
    load_calibration_params_path: str = None


@dataclass
class Parameters:
    save_calibrated_image: bool      = None
    video_calibration_step: int      = None
    num_disparities: int             = None
    block_size: int                  = None
    gaussian_kernel_size: int        = None
    resolution: tuple[int, int]      = None
    alpha: float                     = None
    undistort_method:UndistortMethod = None
