# MADTS
MADTS (**Maritime Auxiliary Detection and Tracking System**) is an auxiliary system for ships that enables the detection and identification of marine traces using pairs of cameras. It also estimates relevant navigation data (heading, distance, speed, friend or foe) making use of stereoscopic vision and displays the data on a tactical HMI.

<img src="./resources/00-MADTS_brief.png" alt="MADTS features">

## Features
* **Vessel detection:** Apply Deep Learning model to detect vessels.
* **Vessel identification:** Classify detected vessel as friend or foe based on its class and behavior.
* **Navigation data estimation:** Estimate key navigation data (heading, distance, speed, friend or foe).
* **Tactical representation:** Display tracks and their estimated navigation data on a tactical Human-Machine Interface.

## Tentative Features
* **Drone detection:** Detect, track and display drones and their navigation data.
* **Tracker:** Track history, track management, track correlation.
* **Enhanced behavior analyzer:** Classify tracks as friend or foe according to their behavior.

## Project Limitations
Due to time, budget, and resource constraints, two *GoPro HERO 10 Black* cameras will be mounted along a radio-controlled warship, which will act as the ownship where **MADTS** will be executed. Other radio-controlled warships and civilian ships will be used for detection and identification.

<img src="./resources/01-MADTS_brief_2.png" alt="MADTS features">

## Architecture
The architecture of MADTS is microservices-based and consists of the following modules:
* **RTPM Servers:** Two RTPM servers that receive the video stream from two *GoPro HERO 10 Black*.
* **VideoSynchronizer:** Uses **OpenGoPro API** to manage and synchronize both video streams.
* **VesselDetector:** Applies Deep Learning to detect and classify vessels.
* **NavDataEstimator:** Estimates navigation data of detected vessels (heading, speed, distance, friend or foe).
* **TrackSender:** Emits track navigation data.
* **Human-Machine Interface:** Displays camera streams and tactical view of detected tracks.

<img src="./resources/02-architecture_diagram.png" alt="MADTS architecture" width="800">

## Modules
TBD. Here goes the description of each module that composes MADTS.

## Communications
TBD. Here goes the desciption of communication between the modules.

## Installation
TBD. Here goes installation instructions.

## Usage
TBD. Here goes usage instructions.

## References
TBD. Here goes citations and references.