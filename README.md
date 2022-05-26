# multiple-grasping-pose-learning
ros package for robot auto data collection and learning different grasping poses for different objects
---
Please see doc/manual.docx

## data collection and learning in PR2

### data collection PR2 look around
launch files
```
roslaunch multiple_grasping_pose_learning pr2_look_around_data_collection_turntable.launch
```
exec eus
```
roscd multiple_grasping_pose_learning/euslisp/
roseus pr2_look_around_collect_data_test.l
(look-around-larm-test)
```
data will saved in `$(find multiple_grasping_pose_learning)/dataset/robot_depth_filter/`


### data collection PR2 tabletop pick&place with teaching
launch files
```
roscd multiple_grasping_pose_learning/launch/pr2/
roslaunch pr2_tabletop_mask.launch
# roslaunch multiple_grasping_pose_learning pr2_tabletop_data_collection_turntable.launch
roslaunch multiple_grasping_pose_learning pr2_data_collection_server_tabletop.launch
```
exec eus
```
roscd multiple_grasping_pose_learning/euslisp/
roseus pr2_tabletop_collect_data_test_with_teaching.l
(short-teach-and-replay-test)
(short-data-test :times 40 :save_img nil :save_avs t)
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

#### その他のデータセット生成プログラム
画像そのままで2次元bboxのアノテーション結果を反映させる．
```
python3 check_annotation_bbox_pr2_look_around.py -t ../dataset/robot_depth_filter/target -c ../dataset/check_data -m multiply_mask
```
rembgとself-filerのかけ合わせたmaskを作って，それでdata augmentationをする．
```
python3 data_augmentation_mask_with_rembg_and_self_pr2_data.py -t ../dataset/rembg_and_mask_with_collision_check/20220311-for-rembg-datas/ -b ../dataset/background/pr2/pr2_20220311/20220311_all/ -a ../dataset/rembg_and_mask_with_collision_check/20220526-test/20220311-for-rembg-datas-bg20220311_all-n10-hb-10-test -n 10 -hb -10
```
これで作ったものは，rgbをtrainとtestにして，そのまま[この手順](https://gist.github.com/fanshi14/cf0b0bc5b6fa1299617c65a46dedd6a7 )の一部に従って学習できる．

### learning from dataset
please edit your `DLBOX_IP` in `train_object_detection_dlbox.sh first`
```
bash train_object_detection_dlbox.sh DATASET_FOLDER
```
