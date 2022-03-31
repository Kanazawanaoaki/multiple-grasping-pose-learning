#!/usr/bin/env roseus

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "multiple_grasping_pose_learning")
(ros::load-ros-manifest "std_srvs")
(ros::roseus "collect_data_turntable")

(format t "init")
;; (ros::wait-for-service "aero_goods_grasping_demo")
(ros::wait-for-service "after_stow_data_collection/save_request")
(ros::advertise "object_name" std_msgs::string 1)

(defun setup()
  (load "package://pr2eus/pr2-interface.l")
  (load "package://multiple_grasping_pose_learning/euslisp/model/red-cup.l")
  (load "package://multiple_grasping_pose_learning/euslisp/pr2_utils.l")
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

;; kanazawa devel
(defun save_data ()
  (let (res)
    (setq req (instance std_srvs::TriggerRequest :init))
    (setq res (ros::service-call "after_stow_data_collection/save_request" req t))
    (format t "data saved! ~%")
    )
  )

(defun save_data_wait ()
  (let (res)
    (setq save-flag nil)
    (until save-flag
           (setq req (instance std_srvs::TriggerRequest :init))
           (setq res (ros::service-call "after_stow_data_collection/save_request" req t))
           (if (send res :success)
               (setq save-flag t)
             ))
    (format t "data saved! ~%")
    )
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
  (send *ri* :speak-jp "物体を握らせて下さい")
  (unix:sleep 2)
  (send *ri* :start-grasp :rarm :wait t)

  ;; (assoc-rcup)
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

(defun defalut-larm ()
  ;; (send *co* :newcoords (make-coords :pos #f(578.018 60.941 1066.015) :rpy #f(-0.839 -0.613 0.499)))
  (send *co* :newcoords (make-coords :pos #f(600 10 1100) :rpy #f(-0.839 -0.613 0.499)))
  (send *robot* :larm :inverse-kinematics
        *co*
        :rotation-axis t
	:revert-if-fail nil)
  (send *ri* :angle-vector (send *robot* :angle-vector) 4000)
  (send *ri* :wait-interpolation)
  )

(defun receive-object-larm ()
  (defalut-larm)

  (send *ri* :stop-grasp :larm :wait t)
  (send *ri* :speak-jp "物体を握らせて下さい")
  (unix:sleep 2)
  (send *ri* :start-grasp :larm :wait t)
  )

(defun return-object-larm ()
  (send *co* :newcoords (make-coords :pos #f(578.018 60.941 1066.015) :rpy #f(-0.839 -0.613 0.499)))
  (send *robot* :larm :inverse-kinematics
        *co*
        :rotation-axis t
	:revert-if-fail nil)
  (send *ri* :angle-vector (send *robot* :angle-vector) 4000)
  (send *ri* :wait-interpolation)


  (send *ri* :speak-jp "物体を受け取って下さい")
  (unix:sleep 2)
  (setq ask-ans '("いいえ"))
  (until (equal ask-ans '("はい"))
	 (send *ri* :speak-jp "物体を離しても良いですか")
	 (dialogue-func)
         (setq ask-ans *voice-data*)
	 )
  (unix:sleep 1)
  (send *ri* :stop-grasp :larm :wait t)
  )

(defun ask-object-name ()
  (setq cnt 0)
  (setq ask-ans '("いいえ"))
  (until (equal ask-ans '("はい"))
         (if (= cnt 0)
             (send *ri* :speak-jp "この物体の名前はなんですか？")
           (send *ri* :speak-jp "もう一度，物体の名前を教えてください"))
         (dialogue-func)
         (setq *obj-name* *voice-data*)
         (send *ri* :speak-jp (format nil "~Aですか？" *obj-name*))
	 (format t "~Aですか？" *obj-name*)
	 (dialogue-func)
         (setq ask-ans *voice-data*)
         (setq cnt (+ cnt 1))
         )
  (setq msg (instance std_msgs::string :init))
  (send msg :data (string (elt *obj-name* 0)))
  (ros::publish "object_name" msg)
  )

(defun ask-end ()
  (let (end-flag)
    (send *ri* :speak-jp "データ収集を終了しますか？")
    (dialogue-func)
    (setq ask-ans *voice-data*)
    (if (equal ask-ans '("はい"))
        (setq end-flag t)
      (setq end-flag nil)
      )
    (send *ri* :speak-jp "了解しました")
    end-flag
    ))

(defun look-around-larm (deg &key(axis :x))
  (send *co* :newcoords (send (send *pr2* :larm :end-coords) :copy-worldcoords))
  (send *co* :rotate (deg2rad deg) axis)
  (send *robot* :larm :inverse-kinematics
        *co*
        :rotation-axis t
        :revert-if-fail nil)
  (send *ri* :angle-vector (send *robot* :angle-vector) 4000)
  (send *ri* :wait-interpolation)
  )

(defun look-around-test ()
  (receive-object)

  ;; (send *ri* :speak-jp "データ収集を開始します")
  (dotimes (i 4)
    (look-around 90)
    (unix:sleep 1);; 念の為
    (save_data)
    )
  ;; (send *ri* :speak-jp "データ収集を修了しました")
  )

(defun look-around-larm-data-collection (&key (speak t))
  (receive-object-larm)

  (if speak
      (send *ri* :speak-jp "データ収集を開始します"))
  (dotimes (i 8)
    (look-around-larm 45)
    (unix:sleep 1);; 念の為
    (save_data)
    )
  (if speak
      (send *ri* :speak-jp "データ収集を修了しました"))
  )

(defun look-around-larm-data-collection-all-with-dialogue ()
  (setq *end-flag* nil)
  (until *end-flag*
         (receive-object-larm)
         (ask-object-name)
         (send *ri* :speak-jp (format nil "~Aのデータ収集を開始します" *obj-name*))
         (dotimes (i 8)
           (look-around-larm 45)
           (unix:sleep 1);; 念の為
           (save_data)
           )
         (send *ri* :speak-jp (format nil "~Aのデータ収集を終了しました" *obj-name*))
         (return-object-larm)
         (setq *end-flag* (ask-end))
         )
  (send *ri* :speak-jp "データ収集を終了します．ご協力ありがとうございました")
  )

(defun look-around-larm-data-collection-all (&key (save_data_flag t))
  (setq *end-flag* nil)
  (setq *obj-cnt* 1)
  (send *ri* :speak-jp "データ収集を開始します")
  (until *end-flag*
         (receive-object-larm)
         ;; (ask-object-name)
         ;; (send *ri* :speak-jp (format nil "~Aのデータ収集を開始します" *obj-name*))
         (send *ri* :speak-jp "物体のデータ収集を開始します")
         (dotimes (i 8)
           (look-around-larm 45)
           (if save_data_flag
               (progn
                 (unix:sleep 1);; 念の為
                 ;; (save_data)
                 (save_data_wait)
                 ))
           )
         ;; (send *ri* :speak-jp (format nil "~Aのデータ収集を終了しました" *obj-name*))

         ;; ここに他の角度も追加したい．
         (dotimes (i 4)
           (look-around-larm -45 :axis :y)
           (if save_data_flag
               (progn
                 (unix:sleep 1);; 念の為
                 ;; (save_data)
                 (save_data_wait)
                 ))
           )
         (defalut-larm)
         (dotimes (i 2)
           (look-around-larm -45 :axis :z)
           (if save_data_flag
               (progn
                 (unix:sleep 1);; 念の為
                 ;; (save_data)
                 (save_data_wait)
                 ))
           )
         (defalut-larm)
         (dotimes (i 2)
           (look-around-larm 45 :axis :z)
           (if save_data_flag
               (progn
                 (unix:sleep 1);; 念の為
                 ;; (save_data)
                 (save_data_wait)
                 ))
           )
         (defalut-larm)

         (send *ri* :speak-jp "物体のデータ収集を終了しました")
         (unix:sleep 2)
         (return-object-larm)
         (setq *end-flag* (ask-end))
         )
  (unix:sleep 1)
  (send *ri* :speak-jp "ご協力ありがとうございました")
  )

(setup)
;; (send *robot* :reset-pose)
(send *robot* :reset-manip-pose)
(send *ri* :angle-vector (send *robot* :angle-vector) 6000)
(send *ri* :wait-interpolation)
(warn "please run (look-around-test) ~%")
(warn "please run (look-around-larm-test) ~%")
