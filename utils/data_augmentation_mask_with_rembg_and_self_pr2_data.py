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

# rgb_data_path = "../dataset/rgb"
# background_data_path = "../dataset/background"
# augmented_data_path = "../dataset/aug_data"
# background_imgs = []

def changedSV(bgr_img, alpha, beta, color_idx):
    hsvimage = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV_FULL) # BGR->HSV
    hsvf = hsvimage.astype(np.float32)
    hsvf[:,:,color_idx] = np.clip(hsvf[:,:,1] * alpha+beta, 0, 255)
    hsv8 = hsvf.astype(np.uint8)
    return cv2.cvtColor(hsv8,cv2.COLOR_HSV2BGR_FULL)


def data_generation(rgbs, masks, labels, countour_points, roi_dimensions, num_objs, num_data, img_counter, hsv_brightness):
    image_counter = img_counter
    for data_id in range(num_data):
        #randomly choose a background
        random_background = np.copy(random.sample(background_imgs,1))[0]
        bg_width = random_background.shape[1]
        bg_height = random_background.shape[0]
        bg_aspect = float(bg_width) / float(bg_height)

        # #set the output image size to 300x300
        # # crop_width = 300
        # # crop_height = 300
        # # crop_width = 640
        # # crop_height = 480
        # crop_height = 300
        # # crop_height = 360
        # crop_width = int(crop_height * bg_aspect) ## align aspect ratio

        # ## random background region
        # height_rand_offset = (bg_height-crop_height) / 2 - 30 + np.random.randint(30)
        # # height_rand_offset = (bg_height-crop_height) / 2 ## w/o random

        # width_rand_offset = 30 + np.random.randint(40)
        # # width_rand_offset = 50 ## w/o random

        # # random_background = random_background[bg_height-crop_height:,bg_width-crop_width:,:] ## w/o random
        # print(bg_height,crop_height,height_rand_offset,bg_width,crop_width,width_rand_offset)
        # print((bg_height-crop_height) / 2 + height_rand_offset,(bg_height+crop_height) / 2 + height_rand_offset, (bg_width-crop_width) / 2 + width_rand_offset,(bg_width + crop_width) / 2 + width_rand_offset)
        # random_background = random_background[int((bg_height-crop_height) / 2 + height_rand_offset) : int((bg_height+crop_height) / 2 + height_rand_offset), int((bg_width-crop_width) / 2 + width_rand_offset) : int((bg_width+crop_width) / 2 + width_rand_offset) , : ]
        # random_background_mask = np.zeros((crop_height,crop_width,3),random_background.dtype)

        ## no crop bg
        crop_width = bg_width
        crop_height = bg_height
        random_background = random_background
        random_background_mask = np.zeros((crop_height,crop_width,3),random_background.dtype)

        #randomly choose n objects
        sample = random.sample(range(0,len(target_obj_rgbs)),num_objs)
        objects = []
        objects_position = []
        for i in range(num_objs):
            rgb_roi = rgbs[sample[i]]
            mask_roi = masks[sample[i]]
            roi_min_y = roi_dimensions[sample[i]][0]
            roi_min_x = roi_dimensions[sample[i]][1]
            roi_height = roi_dimensions[sample[i]][2]
            roi_width = roi_dimensions[sample[i]][3]
            countour_point = countour_points[sample[i]]

            # ratio_x = random.random()
            # ratio_y = random.random()

            ### avoid self-occlusion
            # max_trails = 10 ## try maximum 10 times until no self-occlusion happens
            # max_trails = 200 ## try maximum 200 times until no self-occlusion happens
            max_trails = 1000 ## try maximum 1000 times until no self-occlusion happens
            translate_roi_min_x = translate_roi_min_y = translate_roi_max_x = translate_roi_max_y = 0
            for trail_id in range(0, max_trails):
                # ratio_x = random.random()
                # # ratio_y = random.random()
                # ratio_y = random.random() * 2.0 / 4.0 + 2.0 / 4.0 ## put object only in the lower side of background to simulate the real position

                # ## constraint its position to avoid unfeasible region
                # ## x: 180 - 500, y: 100 - 420 (task-related value)
                # if crop_width == 640:
                #     ratio_x = (ratio_x * (500 - 180) + 180) / float(crop_width)
                #     ratio_y = (ratio_y * (420 - 100) + 100) / float(crop_height)

                # random!!
                ratio_x = random.random()
                ratio_y = random.random()

                translate_roi_min_x = int(ratio_x * crop_width)
                translate_roi_min_y = int(ratio_y * crop_height)

                if translate_roi_min_x + roi_width > crop_width:
                    translate_roi_min_x = crop_width - roi_width
                if translate_roi_min_y + roi_height > crop_height:
                    translate_roi_min_y = crop_height - roi_height

                translate_roi_max_x = translate_roi_min_x + roi_width
                translate_roi_max_y = translate_roi_min_y + roi_height
                if i == 0 or trail_id == max_trails - 1:
                    objects_position.append([translate_roi_min_x, translate_roi_max_x, translate_roi_min_y, translate_roi_max_y])
                    break
                ## check if self occulusion happens
                current_vertices = [[translate_roi_min_x, translate_roi_min_y], [translate_roi_min_x, translate_roi_max_y], [translate_roi_max_x, translate_roi_min_y], [translate_roi_max_x, translate_roi_max_y]]
                self_occulusion = False
                for prev_object_id in range(0, len(objects_position)):
                    for vertice_id in range(0, 4):
                        # if (current_vertices[vertice_id][0] > objects_position[prev_object_id][0] and current_vertices[vertice_id][0] < objects_position[prev_object_id][1] and current_vertices[vertice_id][1] > objects_position[prev_object_id][2] and current_vertices[vertice_id][1] < objects_position[prev_object_id][3]) or (current_vertices[vertice_id][0] < objects_position[prev_object_id][0] and current_vertices[vertice_id][0] > objects_position[prev_object_id][1] and current_vertices[vertice_id][1] < objects_position[prev_object_id][2] and current_vertices[vertice_id][1] > objects_position[prev_object_id][3]):
                        current_cog = [(translate_roi_min_x + translate_roi_max_x) / 2.0, (translate_roi_min_y + translate_roi_max_y) / 2.0]
                        prev_cog = [(objects_position[prev_object_id][0] + objects_position[prev_object_id][1]) / 2.0, (objects_position[prev_object_id][2] + objects_position[prev_object_id][3]) / 2.0]
                        dist_cog = [abs(current_cog[0] - prev_cog[0]), abs(current_cog[1] - prev_cog[1])]
                        current_size = [translate_roi_max_x - translate_roi_min_x, translate_roi_max_y - translate_roi_min_y]
                        prev_size = [objects_position[prev_object_id][1] - objects_position[prev_object_id][0], objects_position[prev_object_id][3] - objects_position[prev_object_id][2]]
                        if dist_cog[0] < (current_size[0] + prev_size[0]) / 2.0 and dist_cog[1] < (current_size[1] + prev_size[1]) / 2.0:
                            self_occulusion = True
                            break
                    if self_occulusion:
                        break
                if self_occulusion:
                    continue
                else:
                    objects_position.append([translate_roi_min_x, translate_roi_max_x, translate_roi_min_y, translate_roi_max_y])
                    break

            # random_background[translate_roi_min_y:translate_roi_max_y,translate_roi_min_x:translate_roi_max_x] = rgb_roi
            transparence = (0,0,0)
            background_img = deepcopy(random_background)
            bg_and_mask_img = deepcopy(random_background)
            bg_and_rgb_img = deepcopy(random_background)
            bg_and_mask_img[translate_roi_min_y:translate_roi_max_y,translate_roi_min_x:translate_roi_max_x] = mask_roi
            bg_and_rgb_img[translate_roi_min_y:translate_roi_max_y,translate_roi_min_x:translate_roi_max_x] = rgb_roi
            random_background = np.where(bg_and_mask_img==transparence, background_img, bg_and_rgb_img)

            random_background_mask[translate_roi_min_y:translate_roi_max_y,translate_roi_min_x:translate_roi_max_x] = mask_roi


            countour_translate_x = roi_min_x - translate_roi_min_x
            countour_translate_y = roi_min_y - translate_roi_min_y
            translated_countour_point = countour_point - np.array([countour_translate_x,countour_translate_y])


            boundingbox =  OrderedDict([
                ("xmin", translate_roi_min_x),
                ("ymin", translate_roi_min_y),
                ("xmax", translate_roi_max_x),
                ("ymax", translate_roi_max_y)
                ])
            obj =  OrderedDict([
                ("name", labels[sample[i]]),
                ("pose", "Unspecified"),
                ("truncated", 0),
                ("difficult", 0),
                ("bndbox",boundingbox)
            ])

            objects.append(obj)

        ## randomize data by adjusting hsv
        hsv_image = cv2.cvtColor(random_background, cv2.COLOR_BGR2HSV)
        hsvf = hsv_image.astype(np.float32)
        hsvf[:,:,2] = np.clip(hsvf[:,:,2] * (1.0 + np.random.randint(hsv_brightness, 10) / 100.0), 0, 255)
        hsv8 = hsvf.astype(np.uint8)
        random_background = cv2.cvtColor(hsv8,cv2.COLOR_HSV2BGR)

        save_rgb = augmented_data_path + "/rgb/frame%04d.jpg" %image_counter
        save_mask = augmented_data_path + "/mask/frame%04d.jpg" %image_counter
        suffix = "frame%04d.jpg" %image_counter
        #path = "/home/himlen/TensorFlow/training_demo/images/frame%04d.jpg" %image_counter
        path = "does not matter"
        image_counter = image_counter + 1
        cv2.imwrite(save_rgb,random_background)
        cv2.imwrite(save_mask,random_background_mask)

        convert_color = cv2.cvtColor(random_background,cv2.COLOR_BGR2RGB)

        imageData = PIL.Image.fromarray(convert_color)
        imageWidth, imageHeight = imageData.size
        # buffer = cStringIO.StringIO()
        buffer = BytesIO.BytesIO()
        imageData.save(buffer, format="JPEG")
        imageData = base64.b64encode(buffer.getvalue())
        source = {'database': "Unknown"}
        data = OrderedDict([
            ("folder","images"),
            ("filename",suffix),
            ("path", path),
            ("source", {'database': "Unknown"}),
            ("size", OrderedDict([("width", crop_width),("height",crop_height),("depth",3)])),
            ("segmented", 0),
            ("object", []),
        ])
        for i in range (len(objects)):
            data["object"].append(objects[i])
        filename = save_rgb.replace("jpg","xml")
        xml = dict2xml(data, wrap = "annotation", indent=" ")
        with open(filename, "w") as f:
            f.write(xml)
        f.close()
    '''
    imageset_path = augmented_data_path + "/ImageSets/"
    total_image = np.arange(image_counter)
    np.random.shuffle(total_image)
    rate = int(image_counter * 0.75)
    train = total_image[:rate]
    val = total_image[rate:]
    train_file_path = imageset_path+"train.txt"
    trainval_file_path = imageset_path+"trainval.txt"
    val_file_path = imageset_path+"val.txt"
    f = open(train_file_path,"w")
    for i in train:
        f.write("frame%03d\n" %i)
    f.close()
    f = open(trainval_file_path,"w")
    for i in total_image:
        f.write("frame%03d\n" %i)
    f.close()
    f = open(val_file_path,"w")
    for i in val:
        f.write("frame%03d\n" %i)
    f.close()
    '''
    return image_counter

def check_and_make_dir(dir_path):
    if False == os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("make dir in path: {}".format(dir_path))

if __name__ == "__main__":
    target_data_path = sys.argv[sys.argv.index("-t") + 1] if "-t" in sys.argv else "../dataset/robot_depth_filter/target"
    background_data_path = sys.argv[sys.argv.index("-b") + 1] if "-b" in sys.argv else "../dataset/background"
    augmented_data_path = sys.argv[sys.argv.index("-a") + 1] if "-a" in sys.argv else "../dataset/aug_data"
    mask_name = sys.argv[sys.argv.index("-m") + 1] if "-m" in sys.argv else "robot_mask"
    aug_obj_num = int(sys.argv[sys.argv.index("-n") + 1]) if "-n" in sys.argv else 3
    hsv_brightness = int(sys.argv[sys.argv.index("-hb") + 1]) if "-hb" in sys.argv else -10
    background_imgs = []
    image_counter=0

    #load background image
    for backgroundfile in glob.glob(os.path.join(background_data_path, '*jpg')):
        background_imgs.append(cv2.imread(backgroundfile))
    for backgroundfile in glob.glob(os.path.join(background_data_path, '*png')):
        background_imgs.append(cv2.imread(backgroundfile))

    # save dataset dir
    save_rgb_dir = augmented_data_path + "/rgb/"
    save_mask_dir = augmented_data_path + "/mask/"
    # save_rembg_dir = augmented_data_path + "/rembg/"
    save_check_dir = augmented_data_path + "/check/"
    # save_rembg_and_mask_dir = augmented_data_path + "/rembg_and_mask/"
    check_and_make_dir(save_rgb_dir)
    check_and_make_dir(save_mask_dir)
    # check_and_make_dir(save_rembg_dir)
    check_and_make_dir(save_check_dir)
    # check_and_make_dir(save_rembg_and_mask_dir)

    target_obj_rgbs = []
    target_obj_masks = []
    target_obj_labels = []
    target_obj_countour_points = []
    target_obj_roi_dimensions = []
    cnt = 0
    label_list = []
    class_txt_file = save_rgb_dir + 'class_names.txt'
    for obj_num in glob.glob(os.path.join(target_data_path, '*')):
        cnt += 1
        label_list.append(str(cnt))
        for datafile in glob.glob(os.path.join(obj_num, '*')):
            for imgfile in glob.glob(os.path.join(datafile, '*jpg')):
                suffix =  imgfile.split("/")[-1]
                if suffix == 'rgb.jpg':
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

                    target_obj_rgbs.append(target_obj)
                    target_obj_masks.append(target_obj_mask)
                    # target_obj_labels.append(obj_num.split("/")[-1])
                    target_obj_labels.append(str(cnt))
                    target_obj_countour_points.append(sampled_point)
                    target_obj_roi_dimensions.append(roi_dimension)

                    ## add for check and original rgb
                    suffix = "frame%04d.jpg" %image_counter
                    save_rgb = save_rgb_dir + suffix
                    save_check = save_check_dir + suffix
                    path = "does not matter"
                    image_counter = image_counter + 1
                    cv2.imwrite(save_rgb,rgb_img)

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

    # # img_counter = data_generation(target_obj_rgbs,target_obj_masks,target_obj_labels,target_obj_countour_points,target_obj_roi_dimensions,1,200,0)
    # # img_counter = data_generation(target_obj_rgbs,target_obj_masks,target_obj_labels,target_obj_countour_points,target_obj_roi_dimensions,2,200,img_counter)
    # # img_counter = data_generation(target_obj_rgbs,target_obj_masks,target_obj_labels,target_obj_countour_points,target_obj_roi_dimensions,3,200,img_counter)
    # # img_counter = data_generation(target_obj_rgbs,target_obj_masks,target_obj_labels,target_obj_countour_points,target_obj_roi_dimensions,4,200,img_counter)
    # # img_counter = data_generation(target_obj_rgbs,target_obj_masks,target_obj_labels,target_obj_countour_points,target_obj_roi_dimensions,5,200,img_counter)

    # # img_counter = data_generation(target_obj_rgbs,target_obj_masks,target_obj_labels,target_obj_countour_points,target_obj_roi_dimensions,1,250,0)
    # # img_counter = data_generation(target_obj_rgbs,target_obj_masks,target_obj_labels,target_obj_countour_points,target_obj_roi_dimensions,2,250,img_counter)
    # # img_counter = data_generation(target_obj_rgbs,target_obj_masks,target_obj_labels,target_obj_countour_points,target_obj_roi_dimensions,3,250,img_counter)
    # # img_counter = data_generation(target_obj_rgbs,target_obj_masks,target_obj_labels,target_obj_countour_points,target_obj_roi_dimensions,4,250,img_counter)

    # img_counter = data_generation(target_obj_rgbs,target_obj_masks,target_obj_labels,target_obj_countour_points,target_obj_roi_dimensions,1,300,0)
    # if max_aug_obj_num >= 2:
    #     img_counter = data_generation(target_obj_rgbs,target_obj_masks,target_obj_labels,target_obj_countour_points,target_obj_roi_dimensions,2,300,img_counter)
    # if max_aug_obj_num >= 3:
    #     img_counter = data_generation(target_obj_rgbs,target_obj_masks,target_obj_labels,target_obj_countour_points,target_obj_roi_dimensions,3,400,img_counter)
    img_counter = data_generation(target_obj_rgbs, target_obj_masks, target_obj_labels, target_obj_countour_points, target_obj_roi_dimensions, aug_obj_num, 1000, image_counter, hsv_brightness)

    # save class_names.text
    f = open(class_txt_file, 'w')
    f.write('_background_\n')
    datalist = []
    for e in label_list:
        datalist.append(e+'\n')
    f.writelines(datalist)
    f.close()
