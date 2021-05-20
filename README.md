# multiple-grasping-pose-learning
ros package for robot auto data collection and learning different grasping poses for different objects
---
## Environment
---
- Ubuntu18.04 and ROS Melodic

## Installation

```bash
git clone https://github.com/himlen1990/multiple-grasping-pose-learning.git
cd multiple-grasping-pose-learning/
catkin -bt
```

## Run

```bash
rosrun multiple_grasping_pose_learning demo
roscd multiple_grasping_pose_learning/euslisp/
roseus collect_data.l
```

## After collected data

```bash
cd utils
python label_generation_labelme_ver.py
create a labels.txt file and add class names
python labelme2voc ../dataset/rgb/01 ../dataset/rgb/voc --labels ../dataset/rgb/01/labels.txt
```
