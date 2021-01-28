class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh
        self.negpos_ratio = neg_pos
        self.device = device

    def forward(self, predictions, targets):
        """
        inputs:
            predictions: SSD出力(taple)
                (loc : [num_batch, 8372,  4],
                conf: [num_batch, 8372, 21],
                dbox_list: [8372, 4])
            targets:
                [num_batch, num_objects, 5]
                5:アノテーション情報[xmin,ymin,xmax,ymax,label]
        outputs:
            loss_l: locの損失
            loss_c: confの損失
        """

        loc_data, conf_data, dbox_list = predictions
        num_batch = loc_data.size(0)  
        num_box = loc_data.size(1)       #8372
        num_classes = conf_data.size(2)  #21

    #各Boxの教師データを作成する
        #各Boxの正解DBoxのラベル格納 num_box分
        conf_t_label = torch.LongTensor(num_batch, num_box).to(self.device)
        #各Boxの正解DBoxの位置格納
        loc_t = torch.LongTensor(num_batch, num_box, 4).to(self.device)
        
        #matchで上書き
        for idx in range(num_batch):
            #正解BBoxとラベル
            truth = targets[idx][:,:-1].to(self.device)
            label = targets[idx][:,-1]to(self.device)

            dbox = dbox_list.to(self.device)
            variance=[0.1,0.2]

            #各Boxの正解データを(loc_t conf_t_labelに格納, IoU>0.5のみを考慮(threshold))
            match(self.jaccard_thresh, truth, dbox, variance, label, loc_t, conf_t_label, idx)

    #背景以外のマスクでloc(Box座標)取り出し
    pos_mask = conf_t_label > 0
    pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)
    #positive
    loc_p = loc_data[pos_idx].view(-1,4)
    loc_t = loc_t[pos_idx].view(-1,4)

    #boxのオフセット情報の損失計算
    loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
    #クラスの損失をクロスエントロピーで計算
    batch_conf = conf_data.view(-1,num_classes)
    loss_c = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction='none')

    #発見したPositiveDBoxの損失を0に
    num_pos = pos_mask.long().sum(1,keepdim=True)
    loss_c = loss_c.view(num_batch, -1)
    loss_c[pos_mask] = 0

    # Hard Negative Miningを実施する
    # 各DBoxの損失の大きさloss_cの順位であるidx_rankを求める
    _, loss_idx = loss_c.sort(1, descending=True)
    _, idx_rank = loss_idx.sort(1)

    #背景のDBoxの数(negative box)を求める
    num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_box)
    neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

    #マスク整形
    pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
    neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

    conf_hnm = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)].view(-1,num_classes)

    conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)]

    loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')

    N = num_pos.sum()
    loss_l /= N
    loss_c /= N
    
    return loss_l, loss_c

def match(threshold, truth, priors, variance, label, loc_t, conf_t, idx):
    """
    各Boxの教師データを作成
        threshold: IoUの閾い値
        truth: データセットの正解Boxデータ(比較対象)
        priors: 予測データ
        variance: オフセット計算時のパラメータ
        label: データセットの正解Labelデータ(比較対象)
        loc_t, conf_t : 教師データを格納するリスト
    """
    overlaps = jaccard( truth, point_form(priors) )
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)

    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j 
    match = truth[best_truth_idx]
    conf = label[best_truth_idx]+1
    #IoUで閾値下回るものは背景(0)
    conf[best_truth_overlap<threshold] = 0
    loc = encode(match, priors, variance)
    loc_t[idx] = loc
    conf_t[idx] = conf

def jaccard(box_a, box_b):
    """
     A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) 正解Box
        box_b: (tensor) 
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:,2]-box_a[:,0]) * (box_a[:,3]-box_a[:,1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:,2]-box_b[:,0]) * (box_b[:,3]-box_b[:,1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter/union

def intersect(box_a, box_b):
    """
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:,2:].unsqueeze(1).expand(A,B,2),
                        box_b[:,2:].unsqueeze(0).expand(A,B,2))
    min_xy = torch.max(box_a[:,:2].unsqueeze(1).expand(A,B,2),
                        box_b[:,:2].unsqueeze(0).expand(A,B,2))                   
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:,:,0]*inter[:,:,1]

def point_form(boxes):
    """ 
    Convert prior_boxes(cx, cy, h, w) to (xmin, ymin, xmax, ymax)
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,    
                     boxes[:, :2] + boxes[:, 2:]/2), 1)

def encode(match, priors, variance):
    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

    """
    #MultiBoxLoss
      オフセットの損失とラベルの損失を求める

      各Boxの教師データを用意しつつ用いるデータを絞り,ロスを計算する.

    #1.jaccared(IoU)を用いた match関数

        関数matchでDBoxから正解BBoxと近いものを抜き出す.
        jaccared(IoU)は0~1の値を取る.

        ●IoUが0.5以上で正解BBoxを持たないDBox
            -> Negative Boxとして正解ラベルを背景(0)とする
            
        ●IoUが0.5以上で正解BBoxを持つDBox
            -> Positive Box としてIoUが最も大きいBoxのラベルを正解データとする
    
    #2.Hard Negative Mining
        Negative Box に分類されたDBoxのうち実際に学習に用いるDBoxの数を絞る.
        8372のDBoxのうち大半がNegative Boxであるため neg/posの比率を決めて絞る
        損失値が高い物を選ぶ.