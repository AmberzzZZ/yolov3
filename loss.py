import keras.backend as K
import tensorflow as tf
import numpy as np
import math


def yolo_loss(args, anchors, n_classes, ignore_thresh=0.5):
    n_layers = len(args) // 2
    y_preds = args[:n_layers]     # [B,H,W,3*(4+1+c)], level0->level2
    y_trues = args[n_layers:]     # [B,H,W,3,4+1+c],   level0->level2

    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]     # big object---> P0
    input_shape = [y_trues[0]._keras_shape[i] * 32 for i in [1,2]]
    grid_shapes = [y_trues[i]._keras_shape[1:3] for i in range(3)]

    loss = 0.
    xy_loss_, wh_loss_, ciou_loss_, conf_loss_, cls_loss_, = 0., 0., 0., 0., 0.
    m = K.shape(y_preds[0])[0]     # batch size
    mf = K.cast(m, K.dtype(y_preds[0]))         # batch size to float
    for i in range(n_layers):
        anchors_l = anchors[anchor_mask[i]]

        conf_gt = y_trues[i][...,4:5]
        cls_gt = y_trues[i][...,5:]

        feats = K.reshape(y_preds[i], (-1, grid_shapes[i][0], grid_shapes[i][1], len(anchors_l), 4+1+n_classes))
        grid_coords, pred_xy, pred_wh, pred_conf, pred_cls = bbox(feats, anchors_l, n_classes, input_shape)     # normed
        pred_box = K.concatenate([pred_xy, pred_wh])       # normed

        conf_gt = y_trues[i][...,4:5]
        cls_gt = y_trues[i][...,5:]
        xy_gt = y_trues[i][...,:2] * grid_shapes[i][::-1] - grid_coords        # offset to grid
        wh_gt = K.log(y_trues[i][...,2:4] * input_shape[::-1] / anchors_l)     # offset to grid
        # wh being too small would cause log(0)=-inf, in this case replace the infs with 0
        wh_gt = K.switch(conf_gt, wh_gt, K.zeros_like(wh_gt))
        box_loss_scale = 2 - y_trues[i][...,2:3]*y_trues[i][...,3:4]

        # box_loss: xy_loss+wh_loss / iou loss
        # xy_loss: bce, based on grid center
        xy_loss = conf_gt * box_loss_scale * K.binary_crossentropy(xy_gt, feats[...,0:2], from_logits=True)
        # wh_loss: l2, based on anchor shape
        wh_loss = conf_gt * box_loss_scale * 0.5 * K.square(wh_gt-feats[...,2:4])
        # ciou_loss: iou
        ciou = tf.expand_dims(bbox_ciou(y_trues[i][...,:4], pred_box, grid_coords, grid_shapes[i], input_shape, anchors_l), axis=-1)
        ciou_loss = conf_gt * box_loss_scale * (1 - ciou)

        # conf_loss: bce
        # ignore mask: objects on gt mask which has iou<ignore_thresh with anchors
        ignore_mask = tf.TensorArray(K.dtype(y_trues[0]), size=1, dynamic_size=True)  # 动态size数组
        object_mask = tf.cast(conf_gt, tf.bool)
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_trues[i][b,...,0:4], object_mask[b,...,0])   # flattened(h*w*a*mask) gt boxes for current sample [N,4]
            iou = box_iou(pred_box[b], true_box)     # [H,W,a,N]
            best_iou = K.max(iou, axis=-1, keepdims=True)     # [H,W,a,1]
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()        # # [b,H,W,a,1]
        conf_loss = conf_gt * K.binary_crossentropy(conf_gt, feats[...,4:5], from_logits=True)+ \
                    (1-conf_gt) * ignore_mask* K.binary_crossentropy(conf_gt, feats[...,4:5], from_logits=True)

        # cls_loss: bce
        cls_loss = conf_gt * K.binary_crossentropy(cls_gt, feats[...,5:], from_logits=True)

        # xy_loss_ += K.sum(xy_loss) / mf
        # wh_loss_ += K.sum(wh_loss) / mf
        ciou_loss_ += K.sum(ciou_loss_) / mf
        conf_loss_ += K.sum(conf_loss) / mf
        cls_loss_ += K.sum(cls_loss) / mf

    # loss = xy_loss_ + wh_loss_ + conf_loss_ + cls_loss_
    loss = ciou_loss_ + conf_loss_ + cls_loss_

    # return loss
    return tf.stack([loss, xy_loss_, wh_loss_, conf_loss_, cls_loss_], axis=0)


# for each level, offset outputs to normed outputs, logit outputs to posiblity outputs
def bbox(feats, anchors, n_classes, input_shape):
    # feats: [B,H,W,3,4+1+c]
    # anchors: anchors for current level, [3,2] arr
    n_anchors = len(anchors)
    anchors_tensor = K.reshape(K.constant(anchors), (1,1,1,n_anchors,2))

    grid_shape = K.int_shape(feats)[1:3]
    h, w = grid_shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    grid_coords = np.stack([x, y], axis=-1).astype(np.float32)
    grid_coords = K.expand_dims(grid_coords, axis=-2)     # expand dim for anchors dim

    box_xy = (K.sigmoid(feats[...,:2]) + grid_coords) / tf.constant(grid_shape[::-1], dtype=tf.float32)
    box_wh = anchors_tensor * K.exp(feats[...,2:4]) / tf.constant(input_shape[::-1], dtype=tf.float32)
    box_conf = K.sigmoid(feats[...,4:5])
    box_cls = K.sigmoid(feats[...,5:])

    return grid_coords, box_xy, box_wh, box_conf, box_cls


# for each level, for each grid, calculate iou with all gt boxes
def box_iou(b1, b2):
    # b1: normed_pred: [H,W,a,4]
    # b2: gt: [N,4]
    # return: [H,W,a,N]
    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)       # [H,W,a,1,4]
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half     # [H,W,a,1,2]

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)        # [1,N,4]
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half      # [1,N,2]

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def bbox_ciou(gt_xywh, pred_xywh, grid_coords, grid_shape, input_shape, anchors):
    # ciou = iou - p2/c2 - av

    # convert to abs xywh    [b,h,w,3,4]
    gt_xy = gt_xywh[...,:2] * input_shape[::-1]
    gt_wh = gt_xywh[...,2:] * input_shape[::-1]
    boxes1 = K.concatenate([gt_xy, gt_wh], axis=-1)
    pred_xy = (pred_xywh[...,:2] + grid_coords) / grid_shape[::-1] * input_shape[::-1]
    pred_wh = pred_xywh[...,2:] * input_shape[::-1]
    boxes2 = K.concatenate([pred_xy, pred_wh], axis=-1)

    # 变成左上角坐标、右下角坐标
    boxes1_x0y0x1y1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_x0y0x1y1 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    # 避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
    boxes1_x0y0x1y1 = tf.concat([tf.minimum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])], axis=-1)
    boxes2_x0y0x1y1 = tf.concat([tf.minimum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])], axis=-1)

    # 两个矩形的面积
    boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
    boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = tf.maximum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    right_down = tf.minimum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-9)

    # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = tf.minimum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    enclose_right_down = tf.maximum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 包围矩形的对角线的平方
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = K.pow(enclose_wh[..., 0], 2) + K.pow(enclose_wh[..., 1], 2)

    # 两矩形中心点距离的平方
    p2 = K.pow(boxes1[..., 0] - boxes2[..., 0], 2) + K.pow(boxes1[..., 1] - boxes2[..., 1], 2)

    # 增加av。加上除0保护防止nan。
    atan1 = tf.atan(boxes1[..., 2] / (boxes1[..., 3] + 1e-9))
    atan2 = tf.atan(boxes2[..., 2] / (boxes2[..., 3] + 1e-9))
    v = 4.0 * K.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v)

    ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
    return ciou




