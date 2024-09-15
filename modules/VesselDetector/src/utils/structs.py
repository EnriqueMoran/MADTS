from dataclasses import dataclass


__author__ = "EnriqueMoran"


@dataclass
class Model:
    model_config_file: str = None
    weights_file:      str = None
    class_names_file:  str = None
    width:  int = None
    height: int = None

@dataclass
class Communication:
    group: str = None
    port:  int = None
    iface: str = None
    ttl:   int = None
    frequency: float = None

@dataclass
class Stream:
    camera: str = None
    record:bool = None
    scale:float = None
    record_path:str   = None
    

@dataclass
class Detection:
    min_confidence:int = None