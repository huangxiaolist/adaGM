import torch.nn as nn
import torch.nn.functional as F


class MaskedSoftmax(nn.Module):
    def __init__(self, dim):
        super(MaskedSoftmax, self).__init__()
        self.dim = dim

    def forward(self, logit, mask=None):
        if mask is None:
            dist = F.softmax(logit, dim=self.dim)
        else:
            # mask the <pad> word's score. then renormalization
            dist_ = F.softmax(logit, dim=self.dim) * mask
            normalization_factor = dist_.sum(self.dim, keepdim=True)
            dist = dist_ / normalization_factor
        return dist