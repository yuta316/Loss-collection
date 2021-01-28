import torch.nn as nn

criterion = nn.BCELoss()
loss = criterion(output, target)

"""
y:target
x:input

l(n) = -w[ y(n) * logx(n) + (1-y(n)) * log(1-x(n)) ]

ターゲットと出力の間のバイナリクロスエントロピーを測定する基準を作成
BCELoss関数では0~1の値に止めるためシグモイド層が必要
"""