from keras.layers import Input, UpSampling2D, concatenate, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from backbone import darknet
from model import fpn_node
from loss import kp_loss
import os


##### custom metrics #####
def xy_loss(y_true, y_pred):
    return K.mean(y_pred[:,1])

def conf_loss(y_true, y_pred):
    return K.mean(y_pred[:,2])

def cls_loss(y_true, y_pred):
    return K.mean(y_pred[:,3])

metric_lst = [xy_loss, conf_loss, cls_loss]


def kp_model(n_classes=24, input_shape=(416,416,3), lr=3e-4, decay=5e-6,
             load_pretrained=False, freeze_body=2, weights_path=''):
    # image input [h,w,c]
    inpt = Input(input_shape)

    # yolo body: darknet53 + fpn
    model_body = yolo_body(input_shape, n_classes)
    if load_pretrained and os.path.exists(weights_path):
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        if freeze_body in [1,2]:
            # Freeze the darknet53 back or freeze all but 3 output layers.
            num = (180, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first %d layers of total %d layers.' % (num, len(model_body.layers)))

    y_pred = model_body(inpt)

    # yt input  [h,w,cls,2+1]
    h, w = input_shape[:2]
    y_true = [Input(shape=(h//{0:32,1:16,2:8}[i],
                           w//{0:32,1:16,2:8}[i],
                           n_classes,
                           2+1)) for i in range(3)]

    # loss layer  [N,4]
    model_loss = Lambda(kp_loss, arguments={'n_classes': n_classes})([*y_pred, *y_true])

    model = Model([inpt, *y_true], model_loss)
    model.compile(Adam(lr=lr, decay=decay),
                  loss=lambda y_true,y_pred: K.mean(y_pred[:,0]),
                  metrics=metric_lst)

    return model


def yolo_body(input_shape, n_classes):
    darknet_back = darknet(input_shape=input_shape, multi_out=True)
    C2, C1, C0 = darknet_back.outputs[-3:]
    U0, P0 = fpn_node(C0, 512, 256, n_classes*(2+1))
    U0 = UpSampling2D(size=2)(U0)
    C1 = concatenate([U0, C1], axis=-1)
    U1, P1 = fpn_node(C1, 256, 128, n_classes*(2+1))
    U1 = UpSampling2D(size=2)(U1)
    C2 = concatenate([U1, C2], axis=-1)
    _, P2 = fpn_node(C2, 128, 0, n_classes*(2+1))

    # reshape
    h0, w0 = K.int_shape(P0)[1:3]
    P0 = Reshape((h0,w0,n_classes,2+1))(P0)
    h1, w1 = K.int_shape(P1)[1:3]
    P1 = Reshape((h1,w1,n_classes,2+1))(P1)
    h2, w2 = K.int_shape(P2)[1:3]
    P2 = Reshape((h2,w2,n_classes,2+1))(P2)

    model = Model(darknet_back.input, [P0, P1, P2])
    return model


if __name__ == '__main__':

    model = kp_model(input_shape=(640,128,3), n_classes=24)
    model.summary()




