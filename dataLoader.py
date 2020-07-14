import numpy as np
import os
import random
import cv2


# img: png
# anno: normed xywh, cls
def dataGenerator(img_dir, anno_dir, batch_size, target_size, anchors, n_classes, max_boxes=20, aug=False):
    img_lst = os.listdir(img_dir)
    while 1:
        random.shuffle(img_lst)
        img_batch = []      # [B,H,W,C]
        yt_batch = []      # [B,N,5]
        for img in img_lst[:batch_size]:
            file_name = img.split('.jpg')[0]
            img = cv2.imread(os.path.join(img_dir, img), 0)
            boxes = np.zeros((max_boxes, 5))
            with open(os.path.join(anno_dir, file_name+'.txt'), 'r') as f:
                for idx, line in enumerate(f.readlines()):
                    if idx >= max_boxes:
                        break
                    if len(line) < 10:
                        continue
                    x, y, w, h, classid = map(float, line.strip().split(" "))
                    boxes[idx] = [x, y, w, h, classid]
            img, boxes = augmentation(img, boxes, target_size, aug=aug)    # [rescale, shift, rotate]->affine, noise
            img_batch.append(img)
            yt_batch.append(boxes)

        img_batch = np.array(img_batch)
        yt_batch = np.array(yt_batch)
        yt_batch_lst = normedPreprocess(yt_batch, target_size, anchors, n_classes)
        print(len(yt_batch_lst))
        yield [img_batch, *yt_batch_lst], np.zeros((batch_size))


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


# map the normed gt boxes to each grid of each feature level
def normedPreprocess(boxes, input_shape, anchors, n_classes):
    # boxes: [B,N,5] normed, 5 for [xywh+clsid]
    # return: [B,H,W,a,5+c] for each level, normed
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]

    input_shape = np.array(input_shape, dtype='int32')     # x,y
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[i] for i in range(3)]

    boxes_wh_abs = boxes[...,2:4] * input_shape      # [B,N,2]
    valid_mask = boxes_wh_abs[..., 0]>0

    batch_size = boxes.shape[0]
    y_true = [np.zeros((batch_size, grid_shapes[l][1], grid_shapes[l][0], 3, 5+n_classes)) for l in range(3)]
    # for each sample
    for b in range(batch_size):
        wh = boxes_wh_abs[b, valid_mask[b]]      # [valid_N, 2]
        if wh.shape[0] == 0:
            continue
        wh = np.expand_dims(wh, -2)      # [valid_N, 1, 2]
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # for each level, for each box, find a best match anchor
        for l in range(3):
            anchors_l = anchors[anchor_mask[l]]
            anchors_l = np.expand_dims(anchors_l, 0)       # [1,3,2]
            anchor_maxes = anchors_l / 2.            # based on [0,0] coords
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors_l[..., 0] * anchors_l[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)     # [valid_N, 3]
            best_anchor = np.argmax(iou, axis=-1)      # [valid_N, 1]

            # generate ground truth for current scale
            for valid_idx, anchor_idx in enumerate(best_anchor):
                if iou[valid_idx, anchor_idx] < 0.6:      # positive threshold
                    continue
                x = int(np.floor(boxes[b, valid_idx, 0] * grid_shapes[l][0]))
                y = int(np.floor(boxes[b, valid_idx, 1] * grid_shapes[l][1]))
                cls_id = int(boxes[b, valid_idx, 4])
                y_true[l][b,y,x,anchor_idx,:4] = boxes[b, valid_idx,:4]
                y_true[l][b,y,x,anchor_idx,4] = 1
                y_true[l][b,y,x,anchor_idx,5+cls_id] = 1

    return y_true


def augmentation(img, boxes, target_size, aug=False):
    # boxes: normed xywh+cls_id, [N,5]
    # rescale: img,xywh
    h,w = img.shape
    img = cv2.resize(img, dsize=target_size)
    factor_y, factor_x = target_size[1]/h, target_size[0]/w
    boxes[:,0:3:2] = boxes[:,0:3:2] * factor_x
    boxes[:,1:4:2] = boxes[:,1:4:2] * factor_y
    # shift: img,xy
    # rotate: img,xywh
    # noise: img
    return img, boxes


if __name__ == '__main__':

    img_dir = "data/img/"
    anno_dir = "data/label/"
    batch_size = 1
    target_size = (256,192)      # x,y
    anchors = get_anchors("yolo_anchors.txt")
    n_classes = 2

    data_generator =  dataGenerator(img_dir, anno_dir, batch_size, target_size,
                                    anchors, n_classes, max_boxes=20)

    for idx, data_batch in enumerate(data_generator):
        input_batch = data_batch[0]
        img_batch, yt_batch = input_batch[0], input_batch[1:]
        print("img: ", img_batch.shape)
        print("yt: ", len(yt_batch))
        print(yt_batch[0].shape)
        print(yt_batch[1].shape)
        print(yt_batch[2].shape)
        if idx > 1:
            break








