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

def dataset_generate(rgb_img,mask_img):
    global image_counter
    save_rgb = save_data_path + "/rgb/frame%04d.jpg" %image_counter
    suffix = "frame%04d.jpg" %image_counter
    save_rembg_rgb = save_data_path + "/rembg/frame%04d.png" %image_counter
    path = "does not matter"
    image_counter = image_counter + 1
    cv2.imwrite(save_rgb,rgb_img)

    img_AND = cv2.bitwise_and(mask_img, rgb_img)
    frame = cv2.cvtColor(img_AND, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(frame)
    png = io.BytesIO() # 空のio.BytesIOオブジェクトを用意
    image.save(png, format='png')
    b_frame = png.getvalue()
    result = remove(b_frame)
    img = PIL.Image.open(io.BytesIO(result)).convert("RGBA")
    img.save(save_rembg_rgb)    

    # ImageFile.LOAD_TRUNCATED_IMAGES = True
    # result = remove(pil_img)
    # img = PIL.Image.open(io.BytesIO(result)).convert("RGBA")
    # img.save(save_rembg_rgb)    

    
if __name__ == "__main__":
    target_data_path = sys.argv[sys.argv.index("-t") + 1] if "-t" in sys.argv else "../dataset/robot_depth_filter/target"
    save_data_path = sys.argv[sys.argv.index("-r") + 1] if "-r" in sys.argv else "../dataset/segmenntation_test/rembg_test"
    mask_name = sys.argv[sys.argv.index("-m") + 1] if "-m" in sys.argv else "robot_mask"
    image_counter = 0

    for obj_num in glob.glob(os.path.join(target_data_path, '*')):
        print(obj_num)
        for datafile in glob.glob(os.path.join(obj_num, '*')):
            for imgfile in glob.glob(os.path.join(datafile, '*jpg')):
                suffix =  imgfile.split("/")[-1]
                if suffix == 'rgb.jpg':
                    rgb_img = cv2.imread(imgfile)
                    maskfile = imgfile.replace("rgb",mask_name)
                    mask_img = cv2.imread(maskfile)
                    # pil_img = np.fromfile(imgfile)
                    dataset_generate(rgb_img,mask_img)
