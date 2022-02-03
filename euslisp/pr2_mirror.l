#!/usr/bin/env roseus

(load "package://multiple_grasping_pose_learning/euslisp/pr2_utils.l")

(unless (boundp '*ri*)
  (progn
    (require :pr2-interface "package://pr2eus/pr2-interface.l")
    (pr2-init)
    (objects (list *pr2*))
    )
  )

(defun initial-pose ()
  (send *pr2* :reset-manip-pose)
  (send *pr2* :l_shoulder_pan_joint :joint-angle 122)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 4000)
  (send *ri* :wait-interpolation)
  (send *viewer* :draw-objects) 
  )

(defun manequine-set ()
  (manequine-mode :arm (list "rarm") :start nil) ;;rarmはマネキンoff (onでも良い？)
  (manequine-mode :arm (list "larm") :start t) ;;larmはマネキンon
  )

(defun manequine-end ()
  (manequine-mode :arm (list "rarm" "larm") :start nil) ;;マネキンoff
  )

(defun mirror (&key (offset 45)) 
  ;; larm の angle-vectorをrarmに移す
  (manequine-end)
  (initial-pose)
  
  ;; マネキンモードのセット
  (manequine-set)
  
  ;; angle-vector mirrorのループ
  (do-until-key
   ;; get now angle-vector
   (send *pr2* :angle-vector (send *ri* :state :potentio-vector))
   (setq lav (send *pr2* :larm :angle-vector))
   
   ;; convert to larm angle-vector (inverts data with index 0,2,4)
   (setq dif-for-adj-vec (float-vector offset 0 0 0 0 0 0))
   (setq n-lav (v- lav dif-for-adj-vec))
   ;; (setq dif-for-invert-vec (scale 2 (float-vector (elt n-lav 0) 0 (elt n-lav 2) 0 (elt n-lav 4) 0 0)))
   (setq dif-for-invert-vec (scale 2 (float-vector (elt n-lav 0) 0 (elt n-lav 2) 0 (elt n-lav 4) 0 (elt n-lav 6))))
   (setq rav (v- n-lav dif-for-invert-vec))
   (format t "lav : ~A , rav : ~A ~%" lav rav)
   ;; (setq inverts-data-list (list 0 2 4))
   ;; (dotimes (i (length lav))
   ;;   (if (= i 0)
   ;; 	 (setq (elt av i) (- (elt lav i) 45))
   ;;     )
   ;;   (if (member i inverts-data-list)
   ;; 	 (setq (elt av i) (* -1 (elt av i)))
   ;;     ))   
   
   ;; send angle-vector
   (send *pr2* :rarm :angle-vector rav)
   (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
   (send *ri* :wait-interpolation nil 0.01)
   )

  (manequine-end)
  )
