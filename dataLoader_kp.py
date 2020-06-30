import numpy as np
import SimpleITK as sitk
import random
import cv2
import os
import json
import math


# bone_ww
HU_min = -500
HU_max = 1000


def dataGenerator(data_dir, mask_limit, target_size, batch_size, n_classes=10,
                  rotate_interval=10, rotate_range=math.pi/6):
    file_lst = [i for i in os.listdir(data_dir) if 'nii' in i and 'seg' not in i]
    while 1:
        random.shuffle(file_lst)
        x_batch = []
        y_batch = []
        for file in file_lst[:batch_size]:
            file_name = file.split('.nii')[0]
            image = sitk.ReadImage(os.path.join(data_dir, file))
            image_arr = sitk.GetArrayFromImage(image)
            # norm: full ww
            # image_arr[image_arr>HU_max] = HU_max
            # image_arr[image_arr<HU_min] = HU_min
            image_arr = (image_arr - np.min(image_arr)) / (np.max(image_arr) - np.min(image_arr))
            # cut
            margin = 20
            zmax, zmin, ymax, ymin, xmax, xmin = mask_limit[file_name]
            zmax, zmin = min(image_arr.shape[0], zmax+margin), max(0, zmin-margin)
            ymax, ymin = min(image_arr.shape[1], ymax+margin), max(0, ymin-margin)
            xmax, xmin = min(image_arr.shape[2], xmax+margin), max(0, xmin-margin)
            image_arr = image_arr[zmin:zmax, ymin:ymax, xmin:xmax]
            # mv kp
            f = open(os.path.join(data_dir, file_name+'_ctd.json'), 'r')
            locs = json.loads(f.read())
            f.close()
            points = {}
            for kp in locs:
                if kp['label']>24:
                    continue
                newx = int(kp['X'] - xmin)
                newy = int(kp['Y'] - ymin)
                newz = int(kp['Z'] - zmin)
                points[newz] = [newx, newy, newz, kp['label']]
            # xy_rotate & mv kp
            rotate_angle = rotate_range / rotate_interval * random.randint(-rotate_interval, rotate_interval+1)
            image_arr, new_points = xy_rotate(image_arr, points, rotate_angle)
            # take yz slice
            yz_mip = np.max(image_arr, axis=2)
            # take xz slice
            xz_mip = np.max(image_arr, axis=1)
            # check
            # for kp in new_points:
            #     cv2.circle(yz_mip, (kp['Y'], kp['Z']), 5, 1, -1)
            # cv2.imshow("tmp", yz_mip)
            # cv2.waitKey(0)
            # resize & mv kp
            x_arr = np.zeros((2,target_size[1],target_size[0],1))
            y_arr = np.zeros((2,target_size[1],target_size[0],2+n_classes+1))
            old_z, old_y = yz_mip.shape
            yz_mip = cv2.resize(yz_mip, target_size)
            x_arr[0,:,:,0] = yz_mip
            for new_p in new_points:
                h = int(math.floor(new_p['Z'] / old_z * target_size[1]))
                w = int(math.floor(new_p['Y'] / old_y * target_size[0]))
                cls = new_p['label']   # [1,n_classes]
                y_arr[0,h,w,0] = new_p['Y'] / old_y       # normed_x
                y_arr[0,h,w,1] = new_p['Z'] / old_z       # normed_y
                y_arr[0,h,w,1+cls] = 1
                y_arr[0,h,w,-1] = 1
            old_z, old_x = xz_mip.shape
            xz_mip = cv2.resize(xz_mip, target_size)
            x_arr[1,:,:,0] = xz_mip
            for new_p in new_points:
                h = int(math.floor(new_p['Z'] / old_z * target_size[1]))
                w = int(math.floor(new_p['X'] / old_x * target_size[0]))
                cls = new_p['label']   # [1,n_classes]
                y_arr[1,h,w,0] = new_p['X'] / old_x       # normed_x
                y_arr[1,h,w,1] = new_p['Z'] / old_z       # normed_y
                y_arr[1,h,w,1+cls] = 1
                y_arr[1,h,w,-1] = 1
            x_batch.append(x_arr)
            y_batch.append(y_arr)
        x_batch = np.concatenate(x_batch, axis=0)
        y_batch = np.concatenate(y_batch, axis=0)

        yield [x_batch, y_batch], np.zeros((y_batch.shape[0],1))


def xy_rotate(arr, points, angle, interpolation=cv2.INTER_LINEAR):
    z,y,x = arr.shape
    rotated_arr = []
    new_points = []
    for i in range(z):      # xy-plane
        slice = arr[i]
        if i in points.keys():         # kp exists in this slice
            rotated_slice, newplst = rotate_img(angle, slice, points=[points[i]], interpolation=interpolation)
            new_p = newplst[0]
            new_points.append({'Z':i, 'Y':int(new_p[1]), 'X':int(new_p[0]), 'label':points[i][-1]})
        else:
            rotated_slice, _ = rotate_img(angle, slice, points=[], interpolation=interpolation)
        rotated_arr.append(np.expand_dims(rotated_slice, axis=0))
    rotated_arr = np.concatenate(rotated_arr, axis=0)
    return rotated_arr, new_points


def rotate_img(angle, img, points=[], interpolation=cv2.INTER_LINEAR):
    h, w = img.shape
    rotateMat = cv2.getRotationMatrix2D((w/2,h/2), math.degrees(angle), 1)
    # img
    rotate_img = cv2.warpAffine(img, rotateMat, (w,h), flags=interpolation, borderValue=(0,0,0))
    # point
    rotated_points = []
    for point in points:
        point = rotateMat.dot([[point[0]], [point[1]], [1]])
        rotated_points.append((int(point[0]), int(point[1])))
    return rotate_img, rotated_points


if __name__ == '__main__':

    data_dir = "/Users/amber/workspace/verse/data"
    target_size = (128, 512)
    batch_size = 1
    mask_limit = {}
    mask_limit['verse536'] = [341,0,266,141,258,160]
    # f = open("maskLimit.txt", 'r')
    # for line in f.readlines():
    #     filename, zmax, zmin, ymax, ymin, xmax, xmin = line.strip().split('\t')
    #     mask_limit[filename] = [int(zmax), int(zmin), int(ymax), int(ymin), int(xmax), int(xmin)]
    # f.close()

    data_generator = dataGenerator(data_dir, mask_limit, target_size, batch_size, n_classes=24,
                                   rotate_interval=10, rotate_range=math.pi/6)

    for idx, [data_batch, y] in enumerate(data_generator):
        x_batch = data_batch[0]
        y_batch = data_batch[1]
        print(x_batch.shape, np.max(x_batch))
        # cv2.imshow("tmp", x_batch[0,:,:,0])
        # cv2.waitKey(0)
        # cv2.imshow("tmp", x_batch[1,:,:,0])
        # cv2.waitKey(0)
        print(y_batch.shape)

        if idx > 0:
            break






















