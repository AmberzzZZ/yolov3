from model import yolo_model
from dataLoader import get_anchors, dataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam


#### custom metrics ####
def xy_loss(y_true, y_pred): return y_pred[1]
def wh_loss(y_true, y_pred): return y_pred[2]
def conf_loss(y_true, y_pred): return y_pred[3]
def cls_loss(y_true, y_pred): return y_pred[4]
metric_lst = [xy_loss, wh_loss, conf_loss, cls_loss]


if __name__ == '__main__':

    img_dir = "data/img/"
    anno_dir = "data/label/"
    n_classes = 1
    target_size = (512,512)      # x,y
    input_shape = (512,512,1)    # h,w
    anchors = get_anchors("prep/yolo_anchors.txt")

    # callbacks
    checkpoint = ModelCheckpoint("weights/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss.3f}.h5",
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # for stage 1
    if True:    # stage 1
        # data
        batch_size = 1
        lr = 1e-3
        train_generator = dataGenerator(img_dir, anno_dir, batch_size, target_size, anchors,
                                        n_classes, max_boxes=10, aug=False)
        val_generator = dataGenerator(img_dir, anno_dir, batch_size, target_size, anchors,
                                      n_classes, max_boxes=10, aug=False)

        # model
        model = yolo_model(anchors, n_classes, input_shape=input_shape, lr=lr,
                           load_pretrained=True, freeze_body=1, weights_path='prep/yolo.h5')
        model.compile(Adam(1e-3), loss=lambda y_true,y_pred: y_pred[0], metrics=metric_lst)

        # train
        steps_per_epoch = 3000//batch_size
        model.fit_generator(generator=train_generator,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_generator,
                            validation_steps=steps_per_epoch//5,
                            epochs=100,
                            initial_epoch=0,
                            callbacks=[checkpoint, reduce_lr, early_stopping],
                            )
        model.save_weights("weights/trained_weights_stage_1.h5")

    # for stage 2
    if False:
        # data
        batch_size = 8
        lr = 5e-4
        train_generator = dataGenerator(img_dir, anno_dir, batch_size, target_size, anchors,
                                        n_classes, max_boxes=10, aug=False)
        val_generator = dataGenerator(img_dir, anno_dir, batch_size, target_size, anchors,
                                      n_classes, max_boxes=10, aug=False)

        # model
        model = yolo_model(anchors, n_classes, input_shape=input_shape, lr=lr, decay=decay,
                           load_pretrained=True, freeze_body=0, weights_path="weights/trained_weights_stage_1.h5")
        model.compile(Adam(5e-4), loss=lambda y_true,y_pred:y_pred[0], metrics=metric_lst)

        # train
        steps_per_epoch = 3000//batch_size
        model.fit_generator(generator=train_generator,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_generator,
                            validation_steps=steps_per_epoch//5,
                            epochs=100,
                            initial_epoch=100,
                            callbacks=[checkpoint, reduce_lr, early_stopping],
                            )
        model.save_weights("weights/trained_weights_final.h5")








