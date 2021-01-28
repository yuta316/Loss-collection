import torch.nn as nn

criterion = nn.MSELoss()
loss = criterion(x, y)

"""
input:x
target:y

1/n * Σ(x(n) - y(n))**2

input,target間のsqrared L2 Norm を計算
L2が平方根をとるのに対し,MSEは平均を取る
"""