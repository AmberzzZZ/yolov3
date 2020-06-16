## darknet
    darknet53: 53层卷积
    darknet block: residual path上1x1 conv + 3x3 conv, 每层卷积都是conv-bn-leaky
    exclude top: 180 layers
    yolo源代码185: 因为它五次下采样多了五层zeropadding


## fpn
    total layers: 247, 和源代码差在那五层zeropadding
    每个level的特征图：level0是backbone的raw output，level1和2是concat(raw output, 上一层up samp)
    shared conv：13131的conv-BN-leaky，1的作用是降维，
    up branch：1x1的conv降低到指定维度
    out branch：3x3的conv + 1x1的conv head


## yt input
    lst, 每个尺度特征图选一个best match anchor(iou最大的), [h,w,a,4+1+c], 
    9选3, 每个gt box在每个尺度都有一个gt anchor offset
    输出三个尺度，output_stride为{0:32,1:16,2:8}，从最顶层的输出开始
    a = 3: anchors对每个尺度提供了三个anchor shape
    4+1+c: [x,y,w,h] + [conf] + [n_classes]
    normed


## y_pred
    lst, 每个尺度对应一组输出, [h,w,a,4+1+c]
    offset


## anchors
    对coco数据集用了9个anchor，kmeans，从小到大，对应检测的level是bottom-up-2-1-0
    [w, h]


## offset(raw output) [t]  --- scale-wise abs [b] --- normed [normed_b]
    bx = sigmoid(tx) + cx
    by = sigmoid(tx) + cy
    cx, cy是某尺度特征图的grid的坐标，bx, by是相对于当前尺度的中心点绝对偏移量，因此normed bx、by也是相对于相应尺度的特征图
    normed_bx = bx / grid_w
    normed_by = by / grid_y
    bw = aw*exp(tw)
    bh = ah*exp(th)
    aw, ah是anchor shape，是相对于原图尺度的绝对值，bw, bh是相对于原图尺度的长宽绝对值，因此normed bw、bh相对于原图尺度
    normed_bx = bx / input_w
    normed_by = by / input_y


## loss
    box_loss_scale: 目标越大权重越小

    loss是在raw offset层面计算的，通过bbox函数计算的normed boxes是为了计算IOU，进而计算ignore_mask

    xy_loss: bce, based on grid center
    针对正样本
    tx,ty用于回归bounding box中心点的位置，gt的中心点落在对应格子内，一定在[0,1]范围内，pred通过sigmoid激活函数限定在[0,1]之间

    wh_loss: l2, based on anchor shape
    针对正样本
    tw,th用于回归bounding box相对于anchor box的尺度，比例无界但是永远大于0，所以pred通过exp函数限定在正数范围内

    conf_loss: bce
    ignore_mask: pos, truth_thresh, ignore, ignore_thresh, neg
    对于正样本（即该格子有目标），计交叉熵。
    对于负样本（即该格子没有目标），只有bbox与ground truth的IOU小于阈值ignore threshold（通常取为0.5），才计交叉熵。

    cls_loss: bce
    针对正样本


## kp model
    做关键点定位的task, back不用动, 把头的bnding box改成xy
    考虑一个格子里面可能有多个点，
    没有宽高不需要anchors，
    所以最后head的输出维度是[H,W,1+n_cls*(2+1)], 
    1 for conf dim, 2 for xy offset, 1 for cls posibility















