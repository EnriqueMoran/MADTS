from dataclasses import dataclass


__author__ = "EnriqueMoran"


@dataclass
class Model:
    model_config_file: str = None
    weights_file:      str = None
    class_names_file:  str = None
    width:  int = None
    height: int = None