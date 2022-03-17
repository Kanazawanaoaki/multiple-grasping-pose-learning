import cv2
import numpy as np
import glob
import os
import base64
import PIL.Image

# from PIL import Image
from PIL import ImageFile
from rembg.bg import remove
import io
from copy import deepcopy

# from io import StringIO
# import io as cStringIO
import io as BytesIO
import random
from collections import OrderedDict
from dict2xml import dict2xml
import sys

def check_and_make_dir(dir_path):
    if False == os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("make dir in path: {}".format(dir_path))

def dataset_generate(rgb_img,pil_img,mask_img,label):
    global image_counter
    suffix = "frame%04d.jpg" %image_counter
    save_rgb = save_rgb_dir + suffix
    save_rembg = save_rembg_dir + "frame%04d.png" %image_counter
    save_check = save_check_dir + suffix
    save_rembg_and_mask = save_rembg_and_mask_dir + "frame%04d.png" %image_counter
    path = "does not matter"
    image_counter = image_counter + 1
    cv2.imwrite(save_rgb,rgb_img)

    # rembg
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    result = remove(pil_img)
    result = remove(result)
    img = PIL.Image.open(io.BytesIO(result)).convert("RGBA")
    img.save(save_rembg)

    # mask
    np_img = np.array(img, dtype=np.uint8)
    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    img_AND = cv2.bitwise_and(mask_img, cv_img)
    cv2.imwrite(save_rembg_and_mask,img_AND)

    # check anotation
    gray = cv2.cvtColor(img_AND, cv2.COLOR_BGR2GRAY)
    img2 = cv2.medianBlur(gray,5)
    ret, thresh = cv2.threshold(img2, 1, 255, 0)
    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    points = np.empty((0,2),int)
    max_contour = contours[0].reshape([-1,2])
    for i in range(len(contours)):
        contour_points = contours[i].reshape([-1,2])
        points= np.append(points,contour_points,axis=0)
    sampled_point = points[::8]
    sorted_point = np.sort(points,axis=0)
    rgb_img = cv2.imread(imgfile)
    min_x = sorted_point[0][0] - np.random.randint(10)
    min_y = sorted_point[0][1] - np.random.randint(10)
    max_x = sorted_point[-1][0] + np.random.randint(10)
    max_y = sorted_point[-1][1] + np.random.randint(10)
    if(min_x<0):
        min_x = 0
    if(min_y<0):
        min_y = 0
    if(max_x>rgb_img.shape[1]):
        max_x = rgb_img.shape[1]
    if(max_y>rgb_img.shape[0]):
        max_y = rgb_img.shape[0]
    height = max_y - min_y
    width  = max_x - min_x
    roi_dimension  = [min_y,min_x,height,width]
    target_obj = rgb_img[min_y:max_y,min_x:max_x]
    target_obj_mask = mask_img[min_y:max_y,min_x:max_x]

    roi_min_y = roi_dimension[0]
    roi_min_x = roi_dimension[1]
    roi_max_y = roi_dimension[0] + roi_dimension[2]
    roi_max_x = roi_dimension[1] + roi_dimension[3]

    check_img = rgb_img
    cv2.rectangle(check_img, pt1=(roi_min_x,roi_max_y), pt2=(roi_max_x,roi_min_y), color=(0,0,255), thickness=2)
    cv2.imwrite(save_check, check_img)

    # rabel
    boundingbox =  OrderedDict([
        ("xmin", roi_min_x),
        ("ymin", roi_min_y),
        ("xmax", roi_max_x),
        ("ymax", roi_max_y)
    ])
    obj =  OrderedDict([
        ("name", label),
        ("pose", "Unspecified"),
        ("truncated", 0),
        ("difficult", 0),
        ("bndbox",boundingbox)
    ])
    data = OrderedDict([
        ("folder","images"),
        ("filename",suffix),
        ("path", path),
        ("source", {'database': "Unknown"}),
        ("size", OrderedDict([("width",int(rgb_img.shape[1])),("height",int(rgb_img.shape[0])),("depth",3)])),
        ("segmented", 0),
        ("object", obj),
    ])
    filename = save_rgb.replace("jpg","xml")
    print(filename)
    xml = dict2xml(data, wrap = "annotation", indent=" ")
    with open(filename, "w") as f:
        f.write(xml)
    f.close()

if __name__ == "__main__":
    target_data_path = sys.argv[sys.argv.index("-t") + 1] if "-t" in sys.argv else "../dataset/robot_depth_filter/target"
    save_data_path = sys.argv[sys.argv.index("-r") + 1] if "-r" in sys.argv else "../dataset/segmenntation_test/rembg_and_mask_test"
    mask_name = sys.argv[sys.argv.index("-m") + 1] if "-m" in sys.argv else "robot_mask"
    image_counter = 0

    save_rgb_dir = save_data_path + "/rgb/"
    save_rembg_dir = save_data_path + "/rembg/"
    save_check_dir = save_data_path + "/check/"
    save_rembg_and_mask_dir = save_data_path + "/rembg_and_mask/"
    check_and_make_dir(save_rgb_dir)
    check_and_make_dir(save_rembg_dir)
    check_and_make_dir(save_check_dir)
    check_and_make_dir(save_rembg_and_mask_dir)

    cnt = 0
    label_list = []
    class_txt_file = save_rgb_dir + 'class_names.txt'
    for obj_num in glob.glob(os.path.join(target_data_path, '*')):
        print(obj_num)
        cnt += 1
        label_list.append(str(cnt))
        for datafile in glob.glob(os.path.join(obj_num, '*')):
            for imgfile in glob.glob(os.path.join(datafile, '*jpg')):
                suffix =  imgfile.split("/")[-1]
                if suffix == 'rgb.jpg':
                    # rgb_img = cv2.imread(imgfile)
                    # maskfile = imgfile.replace("rgb",mask_name)
                    # mask_img = cv2.imread(maskfile)
                    # pil_img = np.fromfile(imgfile)
                    # dataset_generate(rgb_img,pil_img,mask_img,str(cnt))

                    print(imgfile)
                    size = os.path.getsize(imgfile)

                    #rembg
                    pil_img = np.fromfile(imgfile)
                    ImageFile.LOAD_TRUNCATED_IMAGES = True
                    result = remove(pil_img)
                    result = remove(result)
                    img = PIL.Image.open(io.BytesIO(result)).convert("RGBA")

                    # make rembg -> self-filter mask
                    np_img = np.array(img, dtype=np.uint8)
                    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                    maskfile = imgfile.replace("rgb",mask_name)
                    mask_img = cv2.imread(maskfile)
                    mask_img = cv2.bitwise_and(mask_img, cv_img)

                    # apply rembg -> self-filer mask to annotation
                    gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                    img2 = cv2.medianBlur(gray,5)
                    ret, thresh = cv2.threshold(img2, 1, 255, 0)
                    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    points = np.empty((0,2),int)
                    for i in range(len(contours)):
                        contour_points =  contours[i].reshape([-1,2])
                        c_max = np.max(contour_points,axis=0)
                        c_min = np.min(contour_points,axis=0)
                        if (c_max[0] > mask_img.shape[1]*0.2) and (c_max[0] < mask_img.shape[1]*0.8) and (c_max[1] > mask_img.shape[0]*0.2) and (c_max[1] < mask_img.shape[0]*0.8):
                            points= np.append(points,contour_points,axis=0)
                    sampled_point = points[::8]
                    sorted_point = np.sort(points,axis=0)
                    rgb_img = cv2.imread(imgfile)
                    min_x = sorted_point[0][0] - np.random.randint(10)
                    min_y = sorted_point[0][1] - np.random.randint(10)
                    max_x = sorted_point[-1][0] + np.random.randint(10)
                    max_y = sorted_point[-1][1] + np.random.randint(10)
                    print(rgb_img.shape)
                    print(min_x,min_y,max_x,max_y)
                    if(min_x<0):
                        min_x = 0
                    if(min_y<0):
                        min_y = 0
                    if(max_x>rgb_img.shape[1]):
                        max_x = rgb_img.shape[1]
                    if(max_y>rgb_img.shape[0]):
                        max_y = rgb_img.shape[0]

                    height = max_y - min_y
                    width  = max_x - min_x
                    roi_dimension  = [min_y,min_x,height,width]
                    target_obj = rgb_img[min_y:max_y,min_x:max_x]
                    target_obj_mask = mask_img[min_y:max_y,min_x:max_x]

                    # cv2.imshow("sample", target_obj)
                    # cv2.waitKey(0)

                    # target_obj_rgbs.append(target_obj)
                    # target_obj_masks.append(target_obj_mask)
                    # # target_obj_labels.append(obj_num.split("/")[-1])
                    # target_obj_labels.append(str(cnt))
                    # target_obj_countour_points.append(sampled_point)
                    # target_obj_roi_dimensions.append(roi_dimension)

                    ## add for check and original rgb
                    suffix = "frame%04d.jpg" %image_counter
                    save_rgb = save_rgb_dir + suffix
                    save_rembg = save_rembg_dir + "frame%04d.png" %image_counter
                    save_check = save_check_dir + suffix
                    save_rembg_and_mask = save_rembg_and_mask_dir + "frame%04d.png" %image_counter
                    path = "does not matter"
                    image_counter = image_counter + 1
                    cv2.imwrite(save_rgb,rgb_img)
                    img.save(save_rembg)
                    cv2.imwrite(save_rembg_and_mask,mask_img)

                    roi_min_y = roi_dimension[0]
                    roi_min_x = roi_dimension[1]
                    roi_max_y = roi_dimension[0] + roi_dimension[2]
                    roi_max_x = roi_dimension[1] + roi_dimension[3]

                    check_img = deepcopy(rgb_img)
                    cv2.rectangle(check_img, pt1=(roi_min_x,roi_max_y), pt2=(roi_max_x,roi_min_y), color=(0,0,255), thickness=2)
                    cv2.imwrite(save_check, check_img)

                    # rabel
                    boundingbox =  OrderedDict([
                        ("xmin", roi_min_x),
                        ("ymin", roi_min_y),
                        ("xmax", roi_max_x),
                        ("ymax", roi_max_y)
                    ])
                    obj =  OrderedDict([
                        ("name", str(cnt)),
                        ("pose", "Unspecified"),
                        ("truncated", 0),
                        ("difficult", 0),
                        ("bndbox",boundingbox)
                    ])
                    data = OrderedDict([
                        ("folder","images"),
                        ("filename",suffix),
                        ("path", path),
                        ("source", {'database': "Unknown"}),
                        ("size", OrderedDict([("width",int(rgb_img.shape[1])),("height",int(rgb_img.shape[0])),("depth",3)])),
                        ("segmented", 0),
                        ("object", obj),
                    ])
                    filename = save_rgb.replace("jpg","xml")
                    print(filename)
                    xml = dict2xml(data, wrap = "annotation", indent=" ")
                    with open(filename, "w") as f:
                        f.write(xml)
                    f.close()

    f = open(class_txt_file, 'w')
    f.write('_background_\n')
    datalist = []
    for e in label_list:
        datalist.append(e+'\n')
    f.writelines(datalist)
    f.close()
