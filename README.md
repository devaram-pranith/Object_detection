# object_detection
Project Documentation: Object Detection using YOLO (You Only Look Once)
Introduction
This project utilizes YOLO (You Only Look Once), a real-time object detection system, to detect and identify objects in videos. It employs the YOLOv3 architecture, which is known for its speed and accuracy in object detection tasks.
Features
	•	Supports detection of multiple object classes, including persons, cars, bicycles, etc.
	•	Real-time processing of video streams for efficient object detection.
	•	Customizable threshold for confidence level in object detection.
	•	Utilizes pre-trained YOLOv3 weights and configuration files for object detection.
Requirements
	•	Python 3.11
	•	OpenCV (cv2)
	•	NumPy
	•	argparse
Installation
	1	Clone the project repository from GitHub by https://github.com/devaram-pranith/V3_object_detection.git
	2	Install the required dependencies using : pip install opencv-python numpy
Usage
Use Case 1: Person Detection in Video
Code:  python main.py '/path/to/video/file.mp4' "/path/to/yolov3.cfg" "/path/to/yolov3.weights"

Use Case 2: Object Detection in Video with Car and its  Counting
Code: python main1.py --weights '/path/to/yolov3.weights' --cfg '/path/to/yolov3.cfg' --video '/path/to/video/file.mp4'
Use Case 3: All Object Detection in Video with FPS Calculation
Code: python main3.py --weights ‘/path/to/yolov3.weights’ --cfg ‘/path/to/yolov3.cfg’ --video '/path/to/video/file.mp4'


Detected Objects
	•	Use Case 1: Person Detection
	•	Class: Person
	•	Use Case 2: Object Detection with Car Counting
	•	Classes: Car
	•	Use Case 3: General Object Detection
	•	Detected Object Classes:
	•	Person
	•	Bicycle
	•	Car
	•	Motorcycle
	•	Airplane
	•	Bus
	•	Train
	•	Truck
	•	Boat
	•	Traffic light
	•	Fire hydrant
	•	Stop sign
	•	Parking meter
	•	Bench
	•	...
	•	(Complete list of 80 classes available in the code)
Documentation Structure
	1	main.py: Entry point for person detection use case.
	2	main1.py: Entry point for object detection with car counting use case.
	3	main3.py: Entry point for general object detection use case.
	4	Classes: Contains the Yolov3 class for object detection.
	5	Usage: Provides detailed instructions on how to run each use case with command-line arguments.
Note
	•	Ensure that the paths to video files, YOLO configuration, and weights files are correctly specified in the command-line arguments.
	•	For real-time video processing, ensure that the system meets the necessary hardware requirements.
