class OpenPoseLoss(nn.Module):
    def __init__(self):
        super(OpenPoseLoss, self).__init__()

    def forward(self, looses, heatmap_target, heat_mask, paf_target, paf_mask):
        """
        input: 
            losses: Modelの出力
            heatmap_target: 正解の部位のアノテーション  [num_batch,19,46,46]
            heatmap_mask  : heatmap画像のmaks       [num_batch,19,46,46]
            paf_target: [num_batch,38,46,46]
            paf_mask:   [num_batch,38,46,46]
        output:
            loss
        """
        total_loss=0

        #ステージごとに計算
        for j in range(6):
            #PAFsとheatmapにおいてマスクされているところは無視
            #PAFs
            pred1 = losses[2*j]*paf_mask
            gt1 = paf_target.float()*paf_mask

            #heatmaps
            pred2 = losses[2*j+i]*heat_mask
            gt2 = heatmap_target.float().heat_mask

            total_loss+=F.mse_loss(pred1, gt1, reduction='mean') + F.mse_loss(pred2,gt2, reduction='mean')
        return total_loss

criterion = OpenPoseLoss()