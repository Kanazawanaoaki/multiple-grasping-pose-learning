# multiple-grasping-pose-learning
ros package for robot auto data collection and learning different grasping poses for different objects
---
Please see doc/manual.docx

## data collection and learning in PR2

### data collection PR2 look around
```
roscd multiple
roseus 
```


### make dataset from collection data
- 石さんの
```
python data_augmentation_bbox_pr2_look_around.py -t ../dataset/robot_depth_filter/target -b ../dataset/background -a ../dataset/aug_data -n 3
```

- 自分の
```
python annotation_bbox_pr2_look_around.py -t ../dataset/robot_depth_filter/target -g ../dataset/generated_data
```
それで生成されたrgbとラベルのxmlを使って，それをzipする．  
label_map.pbtxtは対応しているものに変更して，  
pipeline.configはコピペする．  

### learning from dataset
please edit your `DLBOX_IP` in `train_object_detection_dlbox.sh first`
```
bash train_object_detection_dlbox.sh DATASET_FOLDER
```
