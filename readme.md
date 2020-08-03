## darknet
    darknet53: 53层卷积
    darknet block: residual path上1x1 conv + 3x3 conv, 每层卷积都是conv-bn-leaky
    exclude top: 180 layers
    yolo源代码185: 因为它五次下采样多了五层zeropadding

    小数据集：initial_filters可以改小
    initial_filters=8: Total params: 2,556,808
    initial_filters=32: Total params: 40,638,496

    # weights convert
    本文的网络结构和darknet.cfg基本一致，除了：
    1. fpn中连conv57之后的两个分支conv59和conv60对调
    2. fpn中连conv66之后的两个分支conv67和conv68对调
    已经在yolo_ref.cfg中改过来了，除此以外，只需要修改三个[yolo]头的n_classes和[net]的长宽通道


## fpn
    total layers: 247, 和源代码差在那五层zeropadding
    每个level的特征图：level0是backbone的raw output，level1和2是concat(raw output, 上一层up samp)
    shared conv：13131的conv-BN-leaky，1的作用是降维，
    up branch：1x1的conv降低到指定维度
    out branch：3x3的conv + 1x1的conv head

    back+fpn: Total params: 61,650,535
    

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
    box_loss_scale: 目标越大权重越小，最终的loss是三个尺度加在一起的

    loss是在raw offset层面计算的，通过bbox函数计算的normed boxes是为了计算IOU，进而计算ignore_mask

    xy_loss: bce, based on grid center
    针对正样本
    tx,ty用于回归bounding box中心点的位置，gt的中心点落在对应格子内，一定在[0,1]范围内，pred通过sigmoid激活函数限定在[0,1]之间

    wh_loss: l2 loss, based on anchor shape
    针对正样本
    tw,th用于回归bounding box相对于anchor box的尺度，比例无界但是永远大于0，所以pred通过exp函数限定在正数范围内

    conf_loss: bce
    ignore_mask: 没有目标的格子中，如果格子预测出物体且预测目标与某gt box的IOU>ignore_thresh, 则视为忽略样本
    通常这部分样本位于gt boxes附近，加入到有无目标的判断中容易混淆。
    对于正样本（即该格子有目标），计交叉熵。
    对于负样本（即该格子没有目标，且预测目标与gt boxes的IOU均小于ignore_thresh），计交叉熵。

    cls_loss: bce
    针对正样本


    [new:]IOU loss
    iou = inter_area / union_area
    ciou = iou - p2/c2 - av, p2是两个box中心点的l2 distance, c2是两个box外接矩形对角线的l2 distance
    


    


## kp model
    做关键点定位的task, back不动, 两个头，一个做xy regression，一个做分类，
    一个GT grid里面可能落多个点，选centerness最大的点作为标签，
    GT label不用点，用类似高斯核的东西，引入centerness替代conf，弱化class imbalance
    没有宽高不需要anchors，没有anchor channel
    reg head输出维度[H,W,2]
    cls head输出维度[H,W,n_cls+1], 1 for centerness
    shared head / separated head: 
        yolo是shared head，xywhclsconf一起出的
        retinaNet是separated head，fpn的特征出来两个head有独立的4个conv+head

    centerness:
    from FCOS, [min(l,r)*min(u,b)] / [max(l,r)*max(u,b)], 可以加个sqrt "to slow down the decay"
    from PolarMask, min(d0,d1,..) / max(d0,d1,...), 
    这两个都是预测点相对contour的距离来计算centerness，
    我们只预测关键点的话，可以直接回归一个高斯核的conf

    loss:
    我们要回归的xy，cls，centerness都是[0,1]之间的值
    base loss可以用bce，
    further more：l2、wce、focal_loss...
























