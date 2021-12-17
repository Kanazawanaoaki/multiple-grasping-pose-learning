#!/usr/bin/env roseus

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "multiple_grasping_pose_learning")
(ros::roseus "collect_data_turntable")

(format t "init")
(ros::wait-for-service "aero_goods_grasping_demo")

(defun setup()
  (load "package://pr2eus/pr2-interface.l")
  (pr2-init)
  (setq *robot* *pr2*)
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
    (format t "image saved ~%")
    ))

(defun setup_table()
  (setq *table_pose* (get_table))
  )
  
(setup)
;; (send *robot* :reset-pose)
(send *robot* :reset-manip-pose)
(send *ri* :angle-vector (send *robot* :angle-vector) 6000)
(send *ri* :wait-interpolation)
