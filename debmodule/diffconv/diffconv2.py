import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class CDConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(CDConv2d, self).__init__()
        assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
        assert kernel_size == 3, 'kernel size for cd_conv should be 3x3'
        assert padding == dilation, 'padding for cd_conv set wrong'
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weights_c = self.weight.sum(dim=[2, 3], keepdim=True)
        yc = F.conv2d(x, weights_c, stride=self.stride, padding=0, groups=self.groups)
        y = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return y - yc


class ADConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ADConv2d, self).__init__()
        assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
        assert kernel_size == 3, 'kernel size for ad_conv should be 3x3'
        assert padding == dilation, 'padding for ad_conv set wrong'
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        shape = self.weight.shape
        weights = self.weight.view(shape[0], shape[1], -1)
        weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape) # clock-wise
        y = F.conv2d(x, weights_conv, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return y


class RDConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(RDConv2d, self).__init__()
        assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
        assert kernel_size == 3, 'kernel size for rd_conv should be 3x3'
        self.stride = stride
        self.padding = 2 * dilation
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        shape = self.weight.shape
        if self.weight.is_cuda:
            buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
        else:
            buffer = torch.zeros(shape[0], shape[1], 5 * 5)
        weights = self.weight.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
        buffer[:, :, 12] = 0
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        y = F.conv2d(x, buffer, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return y



class DEModule(nn.Module):  ###  细节增强模块
    def __init__(self, dim):  ## dim=input dim
        super(DEModule, self).__init__()  # 添加这行
        self.Ref = torch.nn.ReflectionPad2d(1)
        self.conv1 = torch.nn.Conv2d(dim, dim//4, 1, 1, 0)
        # self.conv2 = torch.nn.Conv2d(dim, dim//4, 1, 1, 0)
        self.conv3 = torch.nn.Conv2d(dim, dim//2, 1, 1, 0)
        self.conv4 = torch.nn.Conv2d(dim, dim//2, 3, 1, 1)

        self.CDConv2d = CDConv2d(dim//4, dim//4, 3, padding=1, dilation=1)   ## (in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.ADConv2d = ADConv2d(dim//4, dim//4, 3, padding=1, dilation=1)
        self.RDConv2d = RDConv2d(dim//4, dim//4, 3, padding=1, dilation=1)
        self.act = Mish()

    def forward(self, x):
        conv_x = self.act(self.conv1(x))
        CDConv2d_x = self.act(self.CDConv2d(self.conv1(x)))
        ADConv2d_x = self.act(self.ADConv2d(self.conv1(x)))
        RDConv2d_x = self.act(self.RDConv2d(self.conv1(x)))
        out = torch.cat((conv_x, CDConv2d_x, ADConv2d_x, RDConv2d_x), 1)
        out = self.act(self.conv3(out))

        x = self.act(self.conv4(x))

        out1 = torch.cat((out, x), 1)

        return out1  ## 输出维度保持不变
