from keras.layers import Input, Conv2D, UpSampling2D, concatenate, Lambda
from keras.models import Model
from backbone import darknet, Conv_BN
from loss import yolo_loss
from eval import yolo_eval
from dataLoader import get_anchors
import os


def yolo_model(anchors, n_classes=20, input_shape=(416,416,3), initial_filters=32,
               lr=3e-4, decay=5e-6, eval_model=False, score=.4, iou=.4, max_boxes=20,
               load_pretrained=False, freeze_body=2, weights_path=''):

    # yolo body: darknet53 + fpn
    # input: image input [h,w,c], output: list of [h,w,3*(4+1+c)], 3 for 3 anchors for each level
    n_anchors = len(anchors)
    model_body = yolo_body(input_shape, n_anchors//3, n_classes, initial_filters)
    if load_pretrained and os.path.exists(weights_path):
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        if freeze_body in [1,2]:
            # Freeze the darknet53 back or freeze all but 3 output layers.
            num = (180, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first %d layers of total %d layers.' % (num, len(model_body.layers)))

    y_pred = model_body.outputs

    if not eval_model:
        # yt input  [h,w,a,4+1+c]
        h, w = input_shape[:2]
        y_true = [Input(shape=(h//{0:32,1:16,2:8}[i],
                               w//{0:32,1:16,2:8}[i],
                               n_anchors//3,
                               4+1+n_classes)) for i in range(3)]

        # loss layer
        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                            arguments={'anchors': anchors, 'n_classes': n_classes,
                            'ignore_thresh': 0.5})([*y_pred, *y_true])     # [N,5]

        model = Model([model_body.input, *y_true], model_loss)

    else:    # tobemodified: add eval layer

        model_eval = Lambda(yolo_eval, name="yolo_eval", arguments={'anchors': anchors,
                            'num_classes': n_classes, 'image_shape': input_shape[:2], 'score_threshold': score,
                            'iou_threshold': iou, 'max_boxes': max_boxes})([*y_pred])
        model = Model(model_body.input, model_eval)

    return model


def yolo_body(input_shape, n_anchors, n_classes, initial_filters=32):
    darknet_back = darknet(input_shape=input_shape, multi_out=True, initial_filters=initial_filters)
    C2, C1, C0 = darknet_back.outputs[-3:]
    U0, P0 = fpn_node(C0, 512, 256, n_anchors*(4+1+n_classes))
    U0 = UpSampling2D(size=2)(U0)
    C1 = concatenate([U0, C1], axis=-1)
    U1, P1 = fpn_node(C1, 256, 128, n_anchors*(4+1+n_classes))
    U1 = UpSampling2D(size=2)(U1)
    C2 = concatenate([U1, C2], axis=-1)
    _, P2 = fpn_node(C2, 128, 0, n_anchors*(4+1+n_classes))

    model = Model(darknet_back.input, [P0, P1, P2])
    return model


def fpn_node(x, n_filters, up_filters, out_filters):
    # shared conv
    x = Conv_BN(x, n_filters, 1, strides=1, activation='leaky')
    x = Conv_BN(x, n_filters*2, 3, strides=1, activation='leaky')
    x = Conv_BN(x, n_filters, 1, strides=1, activation='leaky')
    x = Conv_BN(x, n_filters*2, 3, strides=1, activation='leaky')
    x = Conv_BN(x, n_filters, 1, strides=1, activation='leaky')
    # up branch 1x1 conv
    up = Conv_BN(x, up_filters, 1, strides=1, activation='leaky') if up_filters else x
    # out branch 3x3 conv + 1x1 conv head
    out = Conv_BN(x, n_filters*2, 3, strides=1, activation='leaky')
    out = Conv2D(out_filters, 1, strides=1, padding='same')(out)
    return up, out


if __name__ == '__main__':

    anchors = get_anchors('prep/yolo_anchors.txt')
    # model = yolo_model(anchors, input_shape=(512,512,1), initial_filters=32, n_classes=3,
    #                     load_pretrained=True, freeze_body=1, weights_path="convert/yolo.h5")
    # model.summary()

    model = yolo_model(anchors, input_shape=(512,512,1), initial_filters=32, n_classes=3, eval_model=True)

    # model = yolo_body((416,416,3), 9, 20)
    # print(len(model.layers))



