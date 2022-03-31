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

def dataset_generate(rgb_img,pil_img):#mask_img
    global image_counter
    save_rgb = save_data_path + "/rgb/frame%04d.jpg" %image_counter
    suffix = "frame%04d.jpg" %image_counter
    save_rembg_rgb = save_data_path + "/rembg/frame%04d.png" %image_counter
    path = "does not matter"
    image_counter = image_counter + 1
    cv2.imwrite(save_rgb,rgb_img)

    # # crop mask make
    # height = rgb_img.shape[0]
    # width = rgb_img.shape[1]
    # crop_mask = np.zeros((height, width, 3)).astype(np.uint8)
    # cv2.rectangle(crop_mask,
    #               # pt1=(width*0.2, height*0.2),
    #               # pt2=(width*0.8, height*0.8),
    #               pt1=(128, 96),
    #               pt2=(512, 384),
    #               color=(255, 255, 255),
    #               thickness=-1,
    #               lineType=cv2.LINE_4,
    #               shift=0)

    # crop img
    h, w, ch = rgb_img.shape
    img_AND = rgb_img[round(h*0.3):round(h*0.8),round(w*0.25):round(w*0.75),:]

    # img_AND = cv2.bitwise_and(mask_img, rgb_img)
    # img_AND = cv2.bitwise_and(crop_mask, rgb_img)
    frame = cv2.cvtColor(img_AND, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(frame)
    png = io.BytesIO() # 空のio.BytesIOオブジェクトを用意
    image.save(png, format='png')
    b_frame = png.getvalue()
    result = remove(b_frame)
    result = remove(result)
    img = PIL.Image.open(io.BytesIO(result)).convert("RGBA")
    img.save(save_rembg_rgb)

    # ImageFile.LOAD_TRUNCATED_IMAGES = True
    # result = remove(pil_img)
    # result = remove(result)
    # img = PIL.Image.open(io.BytesIO(result)).convert("RGBA")
    # img.save(save_rembg_rgb)


if __name__ == "__main__":
    target_data_path = sys.argv[sys.argv.index("-t") + 1] if "-t" in sys.argv else "../dataset/robot_depth_filter/target"
    save_data_path = sys.argv[sys.argv.index("-r") + 1] if "-r" in sys.argv else "../dataset/segmenntation_test/rembg_test"
    # mask_name = sys.argv[sys.argv.index("-m") + 1] if "-m" in sys.argv else "robot_mask"
    image_counter = 0

    save_rgb_dir = save_data_path + "/rgb/"
    save_rembg_dir = save_data_path + "/rembg/"
    # save_check_dir = save_data_path + "/check/"
    # save_rembg_and_mask_dir = save_data_path + "/rembg_and_mask/"
    check_and_make_dir(save_rgb_dir)
    check_and_make_dir(save_rembg_dir)
    # check_and_make_dir(save_check_dir)
    # check_and_make_dir(save_rembg_and_mask_dir)

    for obj_num in glob.glob(os.path.join(target_data_path, '*')):
        print(obj_num)
        for datafile in glob.glob(os.path.join(obj_num, '*')):
            for imgfile in glob.glob(os.path.join(datafile, '*jpg')):
                suffix =  imgfile.split("/")[-1]
                if suffix == 'rgb.jpg':
                    rgb_img = cv2.imread(imgfile)
                    # maskfile = imgfile.replace("rgb",mask_name)
                    # mask_img = cv2.imread(maskfile)
                    pil_img = np.fromfile(imgfile)
                    dataset_generate(rgb_img,pil_img)#mask_img
