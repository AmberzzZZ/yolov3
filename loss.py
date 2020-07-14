import keras.backend as K
import tensorflow as tf
import numpy as np


def yolo_loss(args, anchors, n_classes, ignore_thresh=0.5, print_loss=False):
    n_layers = len(args) // 2
    y_preds = args[:n_layers]     # [B,H,W,3,4+1+c], level0->level2
    y_trues = args[n_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]     # big object---> P0
    input_shape = [y_trues[0]._keras_shape[i] * 32 for i in [1,2]]
    grid_shapes = [y_trues[i]._keras_shape[1:3] for i in range(3)]

    loss = 0.
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
        xy_gt = y_trues[i][...,:2] * grid_shapes[i][::-1] - grid_coords        # offset
        wh_gt = K.log(y_trues[i][...,2:4] * input_shape[::-1] / anchors_l)     # offset
        # wh being too small would cause log(0)=-inf, in this case replace the infs with 0
        wh_gt = K.switch(conf_gt, wh_gt, K.zeros_like(wh_gt))
        box_loss_scale = 2 - y_trues[i][...,2:3]*y_trues[i][...,3:4]

        # ignore mask: objects on gt mask which has iou<ignore_thresh with anchors
        # iterate over each of batch
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

        # xy_loss: bce, based on grid center
        xy_loss = conf_gt * box_loss_scale * K.binary_crossentropy(xy_gt, feats[...,0:2], from_logits=True)
        # wh_loss: l2, based on anchor shape
        wh_loss = conf_gt * box_loss_scale * 0.5 * K.square(wh_gt-feats[...,2:4])
        # conf_loss: bce
        conf_loss = conf_gt * K.binary_crossentropy(conf_gt, feats[...,4:5], from_logits=True)+ \
                    (1-conf_gt) * ignore_mask* K.binary_crossentropy(conf_gt, feats[...,4:5], from_logits=True)
        # cls_loss: bce
        cls_loss = conf_gt * K.binary_crossentropy(cls_gt, feats[...,5:], from_logits=True)

        xy_loss = K.sum(xy_loss, axis=[1,2,3,4])
        wh_loss = K.sum(wh_loss, axis=[1,2,3,4])
        conf_loss = K.sum(conf_loss, axis=[1,2,3,4])
        cls_loss = K.sum(cls_loss, axis=[1,2,3,4])
        loss += xy_loss + wh_loss + conf_loss + cls_loss

        if print_loss:
            # loss & num of negtives
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, conf_loss, cls_loss, K.sum(ignore_mask)], message='loss: ')

    return K.stack([loss, xy_loss, wh_loss, conf_loss, cls_loss], axis=1)
    # return loss


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


