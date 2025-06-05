import torch
import torch.nn as nn

class LocalEncoder(nn.TransformerEncoder):
    """
    A transformer that applies a local transformation to the input data.
    """

    def __init__(self, encoder_layer, num_layers, mask_size,max_len=2048):
        super(LocalEncoder, self).__init__(encoder_layer, num_layers)

        self.register_buffer("mask", generate_mask(max_len, mask_size), persistent=False)


    def forward(self, x):
        # Apply the local transformation using the model
        return super().forward(x,mask=self.mask[:x.size(0), :x.size(0)])
def generate_mask(length,attention_dist):
    mask=torch.triu(torch.ones((length, length)), diagonal=-attention_dist+1)
    mask=torch.tril(mask, diagonal=0)
    mask=mask.masked_fill(mask!=1, float('-inf'))-1

    return mask
