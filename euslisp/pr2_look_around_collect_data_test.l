#!/usr/bin/env roseus

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "multiple_grasping_pose_learning")
(ros::roseus "collect_data_turntable")

(format t "init")
;; (ros::wait-for-service "aero_goods_grasping_demo")

(defun setup()
  (load "package://pr2eus/pr2-interface.l")
  (load "package://multiple_grasping_pose_learning/euslisp/model/red-cup.l")  
  (pr2-init)
  (setq *robot* *pr2*)
  (setq *co* (make-coords))  
  (setq *rcup* (red-cup))
  (objects (list *robot* *co* *rcup*))
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

(defun save_single_rgb_image()
  (let (res
	get-obj-pose-srv
	obj-pose-res
	)
    
    (setq empty-srv (instance std_srvs::EmptyRequest :init))
    (setq obj-pose-res (ros::service-call
			"aero_new_single_rgb_image_collection"
			empty-srv))
    (format t "image saved")
    ))

(defun setup_table()
  (setq *table_pose* (get_table))
  )


(defun dissoc-rcup ()
  (if (not (eql (send *rcup* :parent) nil))
      (send (send *rcup* :parent) :dissoc *rcup*))
  )

(defun assoc-rcup ()
  (dissoc-rcup)
  (let (local-cds)
    (setq local-cds (make-cascoords))
    (send local-cds :newcoords (send (send *rcup* :handle-handle0) :copy-worldcoords))
    (send local-cds :assoc *rcup*)
    (send local-cds :newcoords (send (send *robot* :rarm :end-coords) :copy-worldcoords))
    (send local-cds :dissoc *rcup*)
    )
  (send *viewer* :draw-objects)
  (send (send *robot* :rarm :end-coords) :assoc *rcup*)  
  )

(defun receive-object ()
  ;; (send *co* :newcoords (make-coords :pos #f(550 -200 900)))
  (send *co* :newcoords (make-coords :pos #f(550 -200 900) :rpy #f(0 0 -1.571)))
  (send *robot* :rarm :inverse-kinematics
	*co*
        :rotation-axis t
	:revert-if-fail nil)
  (send *ri* :angle-vector (send *robot* :angle-vector) 4000)
  (send *ri* :wait-interpolation)
  
  (send *ri* :stop-grasp :rarm :wait t)
  (send *ri* :speak-jp "コップを握らせて下さい")
  (unix:sleep 2)
  (send *ri* :start-grasp :rarm :wait t)
  
  (assoc-rcup)
  )

(defun look-around (deg)
  (send *co* :newcoords (send (send *pr2* :rarm :end-coords) :copy-worldcoords))
  (send *co* :rotate (deg2rad deg) :x)
  (send *robot* :rarm :inverse-kinematics
	*co*
        :rotation-axis t
	:revert-if-fail nil)
  (send *ri* :angle-vector (send *robot* :angle-vector) 4000)
  (send *ri* :wait-interpolation)
  )

(defun set-larm ()
  (send *co* :newcoords (make-coords :pos #f(578.018 60.941 1066.015) :rpy #f(-0.839 -0.613 0.499)))
  (send *robot* :larm :inverse-kinematics
	*co*
        :rotation-axis t
	:revert-if-fail nil)
  (send *ri* :angle-vector (send *robot* :angle-vector) 4000)
  (send *ri* :wait-interpolation)
  )

(defun look-around-larm (deg)
  (send *co* :newcoords (send (send *pr2* :larm :end-coords) :copy-worldcoords))
  (send *co* :rotate (deg2rad deg) :x)
  (send *robot* :larm :inverse-kinematics
	*co*
        :rotation-axis t
	:revert-if-fail nil)
  (send *ri* :angle-vector (send *robot* :angle-vector) 4000)
  (send *ri* :wait-interpolation)
  )

(defun place-object (coords) ;; objectをcoordsに置く
  ;; 置く動作
  (send *co* :newcoords (send coords :copy-worldcoords))
  (send *co* :translate #f(-100 0 100))  
  (send *robot* :rarm :inverse-kinematics
	*co*
	:move-target *rcup*
        :rotation-axis t
	:revert-if-fail nil)
  (send *ri* :angle-vector (send *robot* :angle-vector) 4000)
  (send *co* :translate #f(100 0 -100))  
  (send *robot* :rarm :inverse-kinematics
	*co*
	:move-target *rcup*
        :rotation-axis t
	:revert-if-fail nil)
  (send *ri* :angle-vector (send *robot* :angle-vector) 4000)

  ;; 離す
  (send *ri* :stop-grasp :rarm :wait t)
  (send (send *robot* :rarm :end-coords) :dissoc *rcup*)  
  
  (send *co* :newcoords (send (send *robot* :rarm :end-coords) :copy-worldcoords))
  (send *co* :translate #f(-100 0 0))
  (send *robot* :rarm :inverse-kinematics
	*co*
        :rotation-axis t
	:revert-if-fail nil)
  (send *ri* :angle-vector (send *robot* :angle-vector) 4000)
  
  ;; 初期姿勢に
  (send *robot* :reset-manip-pose)
  (send *ri* :angle-vector (send *robot* :angle-vector) 6000)
  )

(defun pick-object (coords)
  ;; 掴む動作
  (dissoc-rcup)
  (send *rcup* :newcoords (send coords :copy-worldcoords))
  (send *co* :newcoords (send (send *rcup* :handle-handle0) :copy-worldcoords))
  (send *co* :translate #f(-100 0 0))
  (send *robot* :rarm :inverse-kinematics
	*co*
        :rotation-axis t
	:revert-if-fail nil)
  (send *ri* :angle-vector (send *robot* :angle-vector) 4000)
  (send *co* :translate #f(100 0 0))
  (send *robot* :rarm :inverse-kinematics
	*co*
        :rotation-axis t
	:revert-if-fail nil)
  (send *ri* :angle-vector (send *robot* :angle-vector) 4000)

  ;; 離す
  (send *ri* :stop-grasp :rarm :wait t)
  (send (send *robot* :rarm :end-coords) :dissoc *rcup*)  
  
  (send *co* :newcoords (send (send *robot* :rarm :end-coords) :copy-worldcoords))
  (send *co* :translate #f(-100 0 0))
  (send *robot* :rarm :inverse-kinematics
	*co*
        :rotation-axis t
	:revert-if-fail nil)
  (send *ri* :angle-vector (send *robot* :angle-vector) 4000)
  
  ;; 初期姿勢に
  (send *robot* :reset-manip-pose)
  (send *ri* :angle-vector (send *robot* :angle-vector) 6000)  
  )

(setup)
;; (send *robot* :reset-pose)
(send *robot* :reset-manip-pose)
(send *ri* :angle-vector (send *robot* :angle-vector) 6000)
(send *ri* :wait-interpolation)
