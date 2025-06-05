import torch
import torch.nn as nn

#16000
#
#https://github.com/usefulsensors/moonshine/blob/main/moonshine/model.py#L6
class AudioPreprocessor(nn.Module):
    def __init__(self, dim):
        super(AudioPreprocessor, self).__init__()
        # Layer 1: Conv1d with no bias
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=dim, kernel_size=200, stride=100, bias=False,padding=50)
        self.tanh = nn.Tanh()
        # GroupNorm
        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=dim, eps=1e-5)
        # Layer 2: Conv1d, padding=0 corresponds to "valid" in Keras.
        self.conv2 = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=8, stride=4, padding=2)
        self.gelu1 = nn.GELU()
        # Layer 3: Conv1d, padding=0 again corresponds to "valid".
        self.conv3 = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=8, stride=4, padding=2)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)

        x = self.tanh(x)
        x = self.group_norm(x)
        x = self.conv2(x)

        x = self.gelu1(x)
        x = self.conv3(x)
        
        x = self.gelu2(x)
        return x
