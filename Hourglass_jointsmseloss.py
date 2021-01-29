import torch.nn as nn

class JointsMSELoss(nn.Module):
    def __init__(self, use_targt_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_targt_weight = use_targt_weight

    def forward(self, output, target, use_targt_weight):
        batch_size = output.size(0)  #バッチサイズ
        num_joints = output.size(1)  #クラス数
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1,1)
        heatmaps_gt = target.reshape((batch_size, num_jointsm -1)).split(1,1)
        loss=0

        #クラスごとに計算
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            if self.use_targt_weight:
                loss += 0.5 * 
                    self.criterion(heatmaps_pred.mul(target_weight[:,idx]),
                                    heatmaps_gt.mul(target_weight[:,idx]))
            else:
                loss+=0.5*self.criterion(heatmaps_pred, heatmaps_gt)
        return loss/num_joints