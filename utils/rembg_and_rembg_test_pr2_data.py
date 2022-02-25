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

def dataset_generate(rgb_img,pil_img):
    global image_counter
    save_rgb = save_data_path + "/rgb/frame%04d.jpg" %image_counter
    suffix = "frame%04d.jpg" %image_counter
    save_rembg_rgb = save_data_path + "/rembg/frame%04d.png" %image_counter
    path = "does not matter"
    image_counter = image_counter + 1
    cv2.imwrite(save_rgb,rgb_img)

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    result = remove(pil_img)
    result = remove(result)
    img = PIL.Image.open(io.BytesIO(result)).convert("RGBA")
    img.save(save_rembg_rgb)    
    # cv2.rectangle(rembg_img, pt1=(roi_min_x,roi_max_y), pt2=(roi_max_x,roi_min_y), color=(0,0,255), thickness=2)
    # cv2.imwrite(save_rembg_rgb,rembg_img)

    
if __name__ == "__main__":
    target_data_path = sys.argv[sys.argv.index("-t") + 1] if "-t" in sys.argv else "../dataset/robot_depth_filter/target"
    save_data_path = sys.argv[sys.argv.index("-r") + 1] if "-r" in sys.argv else "../dataset/segmenntation_test/rembg_test"
    image_counter = 0

    for obj_num in glob.glob(os.path.join(target_data_path, '*')):
        print(obj_num)
        for datafile in glob.glob(os.path.join(obj_num, '*')):
            for imgfile in glob.glob(os.path.join(datafile, '*jpg')):
                suffix =  imgfile.split("/")[-1]
                if suffix == 'rgb.jpg':
                    rgb_img = cv2.imread(imgfile)
                    pil_img = np.fromfile(imgfile)
                    dataset_generate(rgb_img,pil_img)
