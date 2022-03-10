#!/usr/bin/env roseus

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "std_srvs")

(setq *data* nil)
(ros::roseus "topic_continue")
(defun string-cb (msg)
  (setq *data* (send msg :data))
  (print *data*)
  )

(ros::advertise "object_name_continue" std_msgs::string 1)
(ros::subscribe "object_name" std_msgs::string #'string-cb)

(ros::rate 10)
(do-until-key
 (setq msg (instance std_msgs::string :init))
 (send msg :data *data*)
 (ros::ros-debug "msg [~A]" (send msg :data))
 (ros::publish "object_name_continue" msg)
 (ros::spin-once)
 (ros::sleep)
 )
