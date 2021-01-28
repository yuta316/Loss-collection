loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

"""
smooth L1 (x):
        0.5x^2   (|x|<1)
        |x|-0.5  (oterwise)
"""