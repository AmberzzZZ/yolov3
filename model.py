from keras.layers import Input, Conv2D, UpSampling2D, concatenate, Lambda
from keras.models import Model
import keras.backend as K
from backbone import darknet, Conv_BN
from loss import yolo_loss
from dataLoader import get_anchors
import os


def yolo_model(anchors, n_classes=20, input_shape=(416,416,3),
               load_pretrained=False, freeze_body=2, weights_path=''):
    # image input [h,w,c]
    inpt = Input(input_shape)

    # yolo body: darknet53 + fpn
    n_anchors = len(anchors)
    model_body = yolo_body(input_shape, n_anchors, n_classes)
    if load_pretrained and os.path.exists(weights_path):
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        if freeze_body in [1,2]:
            # Freeze the darknet53 back or freeze all but 3 output layers.
            num = (180, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first %d layers of total %d layers.' % (num, len(model_body.layers)))

    y_pred = model_body(inpt)

    # yt input  [h,w,a,4+1+c]
    h, w = input_shape[:2]
    y_true = [Input(shape=(h//{0:32,1:16,2:8}[i],
                           w//{0:32,1:16,2:8}[i],
                           n_anchors//3,
                           4+1+n_classes)) for i in range(3)]

    # loss layer
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'n_classes': n_classes,
                        'ignore_thresh': 0.5, 'print_loss': False})([*y_pred, *y_true])

    model = Model([inpt, *y_true], model_loss)

    return model


def yolo_body(input_shape, n_anchors, n_classes):
    darknet_back = darknet(input_shape=input_shape, multi_out=True)
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

    anchors = get_anchors('yolo_anchors.txt')
    model = yolo_model(anchors, input_shape=(416,416,3))
    # model.summary()

    # model = yolo_body((416,416,3), 9, 20)
    # print(len(model.layers))



