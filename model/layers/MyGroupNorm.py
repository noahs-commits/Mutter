import torch
import torch.nn as nn

class MyGroupNorm(nn.GroupNorm):
    """
    Custom Group Normalization layer that allows for a variable number of groups.
    """
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, track_running_stats=True):
        super(MyGroupNorm, self).__init__(num_groups, num_channels, eps, affine, track_running_stats)