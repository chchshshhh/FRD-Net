import torch
from torch import nn

class UncertaintyLoss(nn.Module):

    def __init__(self, v_num):
        super(UncertaintyLoss, self).__init__()
        sigma = torch.randn(v_num)
        self.sigma = nn.Parameter(sigma)
        self.v_num = v_num

    def forward(self, *input):
        loss = 0
        for i in range(self.v_num):
            loss += input[i] / (2 * self.sigma[i] ** 2)
        loss += torch.log(self.sigma.pow(2).prod())
        return loss
