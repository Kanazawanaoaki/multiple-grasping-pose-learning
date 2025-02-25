#!/usr/bin/env roseus

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "multiple_grasping_pose_learning")
(ros::roseus "collect_data_turntable")

(format t "init")
(ros::wait-for-service "aero_goods_grasping_demo")

(defun setup()
  (load "package://aeroeus/aero-interface.l")
  (aero-init)
  (setq *robot* *aero*)
  (objects (list *robot*))
  )


(defun get_table();;to get the height of the table, do not put any objects on the table before running this function
  (let (res
	get-table-pose-srv
	obj-pose-res
	pose)
    
    (setq get-obj-pose-srv (instance multiple_grasping_pose_learning::gettableposeRequest :init
				    :command "from euslisp"))
    (setq obj-pose-res (ros::service-call
			"aero_goods_grasping_demo"
			get-obj-pose-srv))
    (setq pose (send obj-pose-res :Pose))
    (format t "~A~%" pose)
    pose
    ))

(defun save_image()
  (let (res
	get-obj-pose-srv
	obj-pose-res
	)
    
    (setq empty-srv (instance std_srvs::EmptyRequest :init))
    (setq obj-pose-res (ros::service-call
			"aero_new_image_collection"
			empty-srv))
    (format t "image saved")
    ))

(defun setup_table()
  (setq *table_pose* (get_table))
  )
  

(defun prepush()
  (let (pre_push_pose)
    (save_image)
    (setq *co* (make-coords))
    (send *co* :translate *table_pose*)
    (send *co* :translate (float-vector -30 -130 80))    
    (send *co* :rotate (deg2rad 45) :z :world)
    (setq pre_push_pose (send *co* :copy-coords))
    (send *robot* :inverse-kinematics
	  *co*
	  :ik-group :rarm-upper-body :rotation-axis t
	  :revert-if-fail nil :look-at-target t)
    (send *ri* :angle-vector (send *robot* :angle-vector) 6000)
    (objects (list *co* *robot*))
    pre_push_pose
  ))

(defun first_push()  
  (send *co* :translate (float-vector -60 100 -20) :world)
  (send *robot* :inverse-kinematics
	*co*
	:ik-group :rarm-upper-body :rotation-axis t
	:revert-if-fail nil :look-at-target t)
  (send *ri* :angle-vector (send *robot* :angle-vector) 6000)
  (unix:usleep(* 4000 1000))
  (send *co* :translate (float-vector -60 100 0) :world)
  (send *robot* :inverse-kinematics
	*co*
	:ik-group :rarm-upper-body :rotation-axis t
	:revert-if-fail nil :look-at-target t)
  (send *ri* :angle-vector (send *robot* :angle-vector) 6000)
  (objects (list *co* *robot*))
  )

(defun second_push()
  (send *co* :translate (float-vector -60 50 0) :world)
  (send *robot* :inverse-kinematics
	*co*
	:ik-group :rarm-upper-body :rotation-axis t
	:revert-if-fail nil :look-at-target t)
  (send *ri* :angle-vector (send *robot* :angle-vector) 6000)
  (unix:usleep(* 8000 1000))
  (send *co* :translate (float-vector 0 100 -20) :world)
  (send *robot* :inverse-kinematics
	*co*
	:ik-group :rarm-upper-body :rotation-axis t
	:revert-if-fail nil :look-at-target t)
  (send *ri* :angle-vector (send *robot* :angle-vector) 6000)
  (unix:usleep(* 5000 1000))
  (send *co* :translate (float-vector 10 100 0) :world)
  (send *robot* :inverse-kinematics
	*co*
	:ik-group :rarm-upper-body :rotation-axis t
	:revert-if-fail nil :look-at-target t)
  (send *ri* :angle-vector (send *robot* :angle-vector) 6000)
  (unix:usleep(* 8000 1000))
  (send *co* :translate (float-vector -60 0 0) :world)
  (send *robot* :inverse-kinematics
	*co*
	:ik-group :rarm-upper-body :rotation-axis t
	:revert-if-fail nil :look-at-target t)
  (send *ri* :angle-vector (send *robot* :angle-vector) 6000)
  )
  
(defun collect_data_once()
  (setq pre_push_pose (prepush))
  (unix:usleep(* 7000 1000))
  (first_push)
  (unix:usleep(* 8000 1000))    
  (send *robot* :reset-pose)
  (send *ri* :angle-vector (send *robot* :angle-vector) 6000)
  (unix:usleep(* 8000 1000))
  (save_image)
  ;;reset  *co*  
  (setq *co* (send pre_push_pose :copy-coords))
  (second_push)
  (unix:usleep(* 8000 1000))
  (send *robot* :reset-pose)
  (send *ri* :angle-vector (send *robot* :angle-vector) 6000)
  (unix:usleep(* 8000 1000))
  (save_image)
  ;;do it again
  (setq *co* (send pre_push_pose :copy-coords))
  (send *co* :translate (float-vector 30 0 0) :world) ;;compansation for the "moving" table
  (send *robot* :inverse-kinematics
	*co*
	:ik-group :rarm-upper-body :rotation-axis t
	:revert-if-fail nil :look-at-target t)
  (send *ri* :angle-vector (send *robot* :angle-vector) 6000)
  (unix:usleep(* 8000 1000))
  (first_push)
  (unix:usleep(* 8000 1000))    
  (send *robot* :reset-pose)
  (send *ri* :angle-vector (send *robot* :angle-vector) 6000)
  (unix:usleep(* 8000 1000))
  (save_image)
  (setq *co* (send pre_push_pose :copy-coords))
  (second_push)
  (unix:usleep(* 8000 1000))
  (send *robot* :reset-pose)
  (send *ri* :angle-vector (send *robot* :angle-vector) 6000)
  (unix:usleep(* 8000 1000))
  (save_image)
  )

(setup)
(send *robot* :reset-pose)
(send *ri* :angle-vector (send *robot* :angle-vector) 6000)
;;(setup_table)
;;(collect_data_once)
