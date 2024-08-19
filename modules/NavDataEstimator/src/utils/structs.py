from dataclasses import dataclass


__author__ = "EnriqueMoran"


@dataclass
class CameraSpecs:
    focal_length: float
    pixel_size: float
    fov: float


@dataclass
class SystemSetup:
    baseline_distance: float


@dataclass
class CalibrationSpecs:
    image_directory: str
    chessboard_width: int
    chessboard_height: int
    chessboard_square_size: int
    frame_width: int
    frame_height: int
    save_calibrated_image: bool
    save_calibration_params_path: str


@dataclass
class Parameters:
    video_calibration_step: int
    num_disparities: int
    block_size: int
    gaussian_kernel_size: int
