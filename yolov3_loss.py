import torch.nn as nn

"""
Boxに関しては二乗誤差(MSE)
Classに関してはCrossEntropy
Objに関しては
"""

mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
obj_scale = 1
noobj_scale = 100

#Boxに関する誤差
loss_x = mse_loss(x[obj_mask], tx[obj_mask])
loss_y = mse_loss(y[obj_mask], ty[obj_mask])
loss_h = mse_loss(w[obj_mask], tw[obj_mask])
loss_w = mse_loss(h[obj_mask], th[obj_mask])
#クラスに関する誤差
loss_cls = bce_loss(pred_cls[obj_mask], tcls[obj_mask])
#object
loss_conf_obj = bce_loss(pred_conf[obj_mask], tconf[obj_mask])
loss_conf_noobj = bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
loss_conf = obj_scale * loss_conf_obj + noobj_scale*loss_conf_noobj

total_loss = loss_x+loss_y+loss_w+loss_h+loss_cls+loss_conf

"""
（参考)以下出力からロスまでの処理

 prediction : 
    [batch_size, num_anchors, size, seize, num_class+5]
"""

    #BCELossにかける前にsigmoidにかける
    x = torch.sigmoid(prediction[..., 0])  # Center x
    y = torch.sigmoid(prediction[..., 1])  # Center y
    w = prediction[..., 2]  # Width
    h = prediction[..., 3]  # Height
    pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
    pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

    # オフセットで修正する
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

    output = torch.cat(
            (   pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ), -1, )
    
    iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

    def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    """
    input:
        pred_boxes : 推測offset Box(cx,xy,w,h)
        pred_cls   : 推測クラス(クラスごとに行う)
        target     : 

    """
    BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)  #Batchサイズ
    nA = pred_boxes.size(1)  #アンカー数
    nC = pred_cls.size(-1)   #クラス数
    nG = pred_boxes.size(2)  #グリッド数

    # アウトプット tensors
    obj_mask = BoolTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = BoolTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # 正解Boxの位置(cx,cy,w,h)
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # IoUを計算し最大のものを取り出す
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf