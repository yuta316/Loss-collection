import torch.nn as nn
criterion = nn.CrossEntropyLoss()
loss = criterion(input, target)

"""
クロスエントロピー
p : input
q : target
H(p,q) = -Σ(p(x)log(q(x)))

真の確率分布pに近づくように導く.
"""