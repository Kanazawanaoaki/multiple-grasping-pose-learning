#!/usr/bin/env roseus

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "multiple_grasping_pose_learning")
(ros::roseus "collect_data_turntable")

(format t "init ~%")
;; (ros::wait-for-service "aero_goods_grasping_demo")
(load "package://multiple_grasping_pose_learning/euslisp/pr2_utils.l")

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
    (format t "image saved ~%")
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
  (send *ri* :speak-jp "物体を握らせて下さい")
  (unix:sleep 3)
  (send *ri* :start-grasp :rarm :wait t)
  
  (assoc-rcup)
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

;; with teaching 
(defun initial-pose-teaching () ;; 道具を握らせて貰って初期姿勢を教わる
  (manequine-end)
  (send *robot* :reset-manip-pose)

  (send *robot* :head-neck-y :joint-angle 0.0)
  (send *robot* :head-neck-p :joint-angle 60.0)
  
  (send *co* :newcoords (make-coords :pos #f(550 -200 900) :rpy #f(0 0 -1.571)))
  (send *robot* :rarm :inverse-kinematics
	*co*
        :rotation-axis t
	:revert-if-fail nil)
  (send *ri* :angle-vector (send *robot* :angle-vector) 4000)
  (send *ri* :wait-interpolation) 

  (send *ri* :stop-grasp :rarm :wait t)
  
  (send *ri* :speak-jp "物体の持ち方を教えて下さい")
  (unix:sleep 2)
  
  ;; okと言われるまで待つ
  (manequine-mode :arm (list "rarm") :start t)
  (setq *ok-flag* nil)
  (until *ok-flag*
	 (dialogue-func)
	 (if (equal *voice-data* '("OK"))
	     (progn
	       (format t "了解しました！ ~%" *voice-data*)
	       (setq *ok-flag* t)
	       )
	   (format t "data: ~A ~%" *voice-data*)
	   ))  
  
  (send *ri* :speak-jp "ありがとうございます")
  (manequine-end)
  (send *robot* :head-neck-y :joint-angle 0.0)
  (send *robot* :head-neck-p :joint-angle 60.0)
  (send *robot* :angle-vector (send *ri* :state :potentio-vector))
  (setq *initial-av* (send *robot* :angle-vector))
  (unix:sleep 2)
  (send *ri* :start-grasp :rarm :wait t) ;; TODO 持つ強さ？
  
  ;; TODO この物体は何ですか？
  )

(defun voice-ok-cb (msg)
  (when msg
    (setq *msg* msg)
    (setq *data* (send *msg* :transcript))
    (if (equal *data* '("OK"))
        (progn
          (format t "了解しました！ ~%" *data*)
          (setq *ok-flag* t)
          )
      (format t "data: ~A ~%" *data*)
      ) 
    )
  )

(defun set-initial-av ()
  (send *robot* :angle-vector *initial-av*)
  (send *ri* :angle-vector (send *robot* :angle-vector) 4000)
  (send *ri* :wait-interpolation) 
  )

(defun replay-rav-list (rav-list &key (time 200))
  (setq rav-length (length rav-list))
  (format t "length of rav-list : ~A ~%" rav-length)
  (send *robot* :angle-vector (send *ri* :state :potentio-vector))
  (setq now-head-y (send *robot* :head-neck-y :joint-angle))
  (setq now-head-p (send *robot* :head-neck-p :joint-angle))
  
  (setq avs (list))
  (setq tms (list))
  (dolist (av rav-list)
    ;; (send *robot* :head-neck-y :joint-angle now-head-y)
    ;; (send *robot* :head-neck-p :joint-angle now-head-p)
    ;; (send *robot* :head-neck-y :joint-angle -0.005299)
    ;; (send *robot* :head-neck-p :joint-angle 52.746)
    (send *robot* :head-neck-y :joint-angle 0.0)
    (send *robot* :head-neck-p :joint-angle 60.0)
    (send *robot* :rarm :angle-vector av)
    (setq avs (append avs (list (send *robot* :angle-vector))))
    (setq tms (append tms (list time)))
    )
  (send *ri* :angle-vector-sequence avs tms)
  (send *ri* :wait-interpolation)  
  )

(defun record-rav-key ()
  (manequine-mode :arm (list "rarm") :start t)
  (setq *ok-flag* nil)
  (setq  rav-list (list))
  (do-until-key
   (send *robot* :angle-vector (send *ri* :state :potentio-vector))
   (setq now-rav (send *robot* :rarm :angle-vector))
   (format t "now rav : ~A" now-rav)
   (setq rav-list (append rav-list (list now-rav)))
   (unix:usleep 100000)
   )
  (manequine-end)
  (setq rav-length (length rav-list))
  (format t "length of rav-list : ~A ~%" rav-length)
  rav-list
  )

(defun record-rav-voice ()
  ;; okと言われるまで待つ
  (manequine-mode :arm (list "rarm") :start t)
  (ros::unsubscribe "/speech_to_text")
  (setq *ok-flag* nil)
  (ros::subscribe "/speech_to_text" speech_recognition_msgs::SpeechRecognitionCandidates #'voice-ok-cb)
  (setq rav-list (list))
  (setq rec-list (list))
  (ros::rate 10)
  (until *ok-flag*
	 (send *robot* :angle-vector (send *ri* :state :potentio-vector))
	 (setq now-rav (send *robot* :rarm :angle-vector))
	 (setq now-rec (send (send *robot* :rarm :end-coords) :copy-worldcoords))
	 (format t "now rav : ~A" now-rav)
	 (setq rav-list (append rav-list (list now-rav)))
	 (setq rec-list (append rec-list (list now-rec)))
	 ;; todo also save times ?
         (ros::spin-once)
         (ros::sleep)
	 )
  
  (send *ri* :speak-jp "ありがとうございます")
  (manequine-end)
  (setq rav-length (length rav-list))
  (format t "length of rav-list : ~A ~%" rav-length)
  (list rav-list rec-list)
  )

(defun initial-pose-teaching-key () ;; 道具を握らせて貰って初期姿勢を教わる
  (manequine-end)
  (send *robot* :reset-manip-pose)
  (send *co* :newcoords (make-coords :pos #f(550 -200 900) :rpy #f(0 0 -1.571)))
  (send *robot* :rarm :inverse-kinematics
	*co*
        :rotation-axis t
	:revert-if-fail nil)
  (send *ri* :angle-vector (send *robot* :angle-vector) 4000)
  (send *ri* :wait-interpolation) 

  (send *ri* :stop-grasp :rarm :wait t)
  
  ;; 待つ
  (manequine-mode :arm (list "rarm") :start t)
  (do-until-key
   (send *robot* :angle-vector (send *ri* :state :potentio-vector))
   (setq now-rav (send *robot* :rarm :angle-vector))
   (format t "now rav : ~A" now-rav)
   (unix:usleep 100000)
   )
  (manequine-end)
  (send *robot* :angle-vector (send *ri* :state :potentio-vector))
  (setq *initial-av* (send *robot* :angle-vector))
  (send *ri* :start-grasp :rarm :wait t) ;; TODO 持つ強さ？  
  )

(defun place-object-teaching ()
  (set-initial-av)

  ;; 物体の置き方を教わる
  (send *ri* :speak-jp "物体の置き方を教えて下さい")
  (unix:sleep 2)

  ;; ;; okと言われるまで待つ
  ;; (manequine-mode :arm (list "rarm") :start t)
  ;; (ros::unsubscribe "/speech_to_text")
  ;; (setq *ok-flag* nil)
  ;; (ros::subscribe "/speech_to_text" speech_recognition_msgs::SpeechRecognitionCandidates #'voice-ok-cb)
  ;; (setq  *place-rav-list* (list))
  ;; (ros::rate 10)
  ;; (until *ok-flag*
  ;; 	 (send *robot* :angle-vector (send *ri* :state :potentio-vector))
  ;; 	 (setq *now-rav* (send *robot* :rarm :angle-vector))
  ;; 	 (format t "now rav : ~A" *now-rav*)
  ;; 	 (setq *place-rav-list* (append *place-rav-list* (list *now-rav*)))
  ;; 	 ;; todo also save times ?
  ;;        (ros::spin-once)
  ;;        (ros::sleep)
  ;; 	 )
  
  ;; (send *ri* :speak-jp "ありがとうございます")
  ;; (manequine-end)
  ;; (print *place-rav-list*)

  ;; (setq *place-rav-list* (record-rav-voice))  
  (setq place-ans (record-rav-voice))  
  (setq *place-rav-list* (elt place-ans 0))  
  (setq *place-rec-list* (elt place-ans 1))  
  
  (send *ri* :stop-grasp :rarm :wait t);; TODO 離す時のグリッパの離し具合？

  ;; 手の離し方
  (send *ri* :speak-jp "手の離し方を教えて下さい")
  (unix:sleep 2)

  ;; okと言われるまで待つ
  ;; (manequine-mode :arm (list "rarm") :start t)
  ;; (ros::unsubscribe "/speech_to_text")
  ;; (setq *ok-flag* nil)
  ;; (ros::subscribe "/speech_to_text" speech_recognition_msgs::SpeechRecognitionCandidates #'voice-ok-cb)
  ;; (setq  *relase-rav-list* (list))
  ;; (ros::rate 10)
  ;; (until *ok-flag*
  ;; 	 (send *robot* :angle-vector (send *ri* :state :potentio-vector))
  ;; 	 (setq *now-rav* (send *robot* :rarm :angle-vector))
  ;; 	 (setq *now-rav* (send *robot* :rarm :end-coords))
  ;; 	 (format t "now rav : ~A" *now-rav*)
  ;; 	 (setq *release-rav-list* (append *release-rav-list* (list *now-rav*)))
  ;; 	 ;; todo also save times ?
  ;;        (ros::spin-once)
  ;;        (ros::sleep)
  ;; 	 )

  ;; (send *ri* :speak-jp "ありがとうございます")
  ;; (manequine-end)
  ;; (print *release-rav-list*)

  ;; (setq *release-rav-list* (record-rav-voice))  
  (setq release-ans (record-rav-voice))  
  (setq *release-rav-list* (elt release-ans 0))  
  (setq *release-rec-list* (elt release-ans 1))  

  (set-initial-av)
  )

(defun pick-object-teaching ()
  (set-initial-av)

  ;; 物体の掴み方を教わる
  (send *ri* :speak-jp "物体の掴み方を教えて下さい")
  (unix:sleep 2)
  
  ;; okと言われるまで待つ
  ;; (manequine-mode :arm (list "rarm") :start t)
  ;; (ros::unsubscribe "/speech_to_text")
  ;; (setq *ok-flag* nil)
  ;; (ros::subscribe "/speech_to_text" speech_recognition_msgs::SpeechRecognitionCandidates #'voice-ok-cb)
  ;; (setq  *pick-rav-list* (list))
  ;; (ros::rate 10)
  ;; (until *ok-flag*
  ;; 	 (send *robot* :angle-vector (send *ri* :state :potentio-vector))
  ;; 	 (setq *now-rav* (send *robot* :rarm :angle-vector))
  ;; 	 (format t "now rav : ~A" *now-rav*)
  ;; 	 (setq *pick-rav-list* (append *pick-rav-list* (list *now-rav*)))
  ;; 	 ;; todo also save times ?
  ;;        (ros::spin-once)
  ;;        (ros::sleep)
  ;; 	 )
  
  ;; (send *ri* :speak-jp "ありがとうございます")
  ;; (manequine-end)
  ;; (print *pick-rav-list*)

  ;; (setq *pick-rav-list* (record-rav-voice))  
  (setq pick-ans (record-rav-voice))  
  (setq *pick-rav-list* (elt pick-ans 0))  
  (setq *pick-rec-list* (elt pick-ans 1))  
  
  (send *ri* :start-grasp :rarm :wait t)
  (unix:sleep 1)
  ;; TODO 持ち上げる？
  
  (send *ri* :stop-grasp :rarm :wait t);; TODO 離す時のグリッパの離し具合？
  (unix:sleep 2)
  
  ;; 離し方を実行する
  (replay-rav-list *release-rav-list*)
  
  (set-initial-av)
  )

(defun teaching-test ()
  ;; 初期姿勢，持ち方の教示
  (initial-pose-teaching)

  ;; 物体の置き方と手の離し方の教示
  (place-object-teaching)

  ;; 物体の掴み方の教示
  (pick-object-teaching)

  (warn "please run (replay-test) ~%")
  )

(defun replay-test ()
  ;; ここで机の状態を見る
  
  ;; 道具を持たせて貰う
  (receive-object)

  ;; 道具を置く
  (set-initial-av)
  (send *ri* :start-grasp :rarm :wait t)
  (unix:sleep 2)  
  (replay-rav-list *place-rav-list*)

  ;; 道具を離す
  (send *ri* :stop-grasp :rarm :wait t)
  (unix:sleep 2)
  (replay-rav-list *release-rav-list*)
  
  ;; ここで道具を見て，バウンディングボックスとか色々な物を保存する．
  
  ;; 道具を掴む
  (set-initial-av)
  (send *ri* :stop-grasp :rarm :wait t)
  (unix:sleep 2)
  (replay-rav-list *pick-rav-list*)
  
  (send *ri* :start-grasp :rarm :wait t)
  (unix:sleep 2)  
  
  (warn "please run (replay-test) ~%")
  )

(defun now-test ()
  ;; 動作を教わる
  (teaching-test)
  
  ;; 教わった動作を再生する
  (send *ri* :speak-jp "教わった動作を実行します")
  (replay-test)
  
  (send *robot* :rarm :move-end-pos #f(0 0 100) :world)
  (send *ri* :angle-vector (send *robot* :angle-vector))
  (send *ri* :wait-interpolation)

  (send *ri* :speak-jp "動作を実行しました")
  )

(defun tmp-data-collection (&key (times 10))
  ;; 机の上にそのまま物を置いていく
  ;; 道具を持たせて貰う
  (receive-object)

  (dotimes (i times)
    ;; 物体を置く
    (set-initial-av)
    (send *ri* :start-grasp :rarm :wait t)
    
    ;; 物体を置く場所を計画する．平面の中でどこに置くか
    
    (start-data-collection :service "place_data_collection_rosbag/start_request");; 置く時のデータ収集
    (unix:sleep 4)  
    (replay-rav-list *place-rav-list*)
    (end-data-collection :service "place_data_collection_rosbag/end_request");; 置く時のデータ収集

    ;; 道具を離す
    (send *ri* :stop-grasp :rarm :wait t)
    (unix:sleep 2)
    (replay-rav-list *release-rav-list*)
    
    ;; ここで道具を見て，バウンディングボックスとか色々なdataを保存する．
        
    ;; 道具を掴む
    (set-initial-av)
    (send *ri* :stop-grasp :rarm :wait t)
    
    (start-data-collection :service "pick_data_collection_rosbag/start_request");; 掴む時のデータ収集
    (unix:sleep 4)
    (replay-rav-list *pick-rav-list*)
    (end-data-collection :service "pick_data_collection_rosbag/end_request");; 掴む時のデータ収集
    
    (send *ri* :start-grasp :rarm :wait t)
    (unix:sleep 2)

    (send *robot* :rarm :move-end-pos #f(0 0 100) :world)
    (send *ri* :angle-vector (send *robot* :angle-vector))
    (send *ri* :wait-interpolation)
    ) 
  )

(defun tmp-test ()
  (initial-pose-teaching-key)

  (set-initial-av)

  (setq *tmp-rav-list* (record-rav-key))
  (set-initial-av)
  (replay-rav-list *tmp-rav-list* :time 200)
  (set-initial-av)
  )

(defun tmp-test-voice ()
  (initial-pose-teaching)
  
  (set-initial-av)
  
  ;; (setq *tmp-rav-list* (record-rav-voice))
  (setq tmp-ans (record-rav-voice))  
  (setq *tmp-rav-list* (elt tmp-ans 0))  
  (setq *tmp-rec-list* (elt tmp-ans 1))  

  (set-initial-av)
  (replay-rav-list *tmp-rav-list* :time 200)
  (set-initial-av)
  )


(manequine-end)
(setup)
;; (send *robot* :reset-pose)
(send *robot* :reset-manip-pose)
(send *ri* :angle-vector (send *robot* :angle-vector) 6000)
(send *ri* :wait-interpolation)
