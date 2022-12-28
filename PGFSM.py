import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn

class SSPFM(Module):
    def __init__(self, in_dim):
        super(SSPFM, self).__init__()

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.alpha_1 = Parameter(torch.zeros(1))
        self.beta_1 = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        # B,C,H,W -> B,1,HW
        a = self.value_conv(x)
        a_max, l = torch.max(a, dim=1, keepdim=True)
        a_max = a_max.view(m_batchsize, -1, width*height)
        a_mean = torch.mean(a, dim=1, keepdim=True).view(m_batchsize, -1, width*height)
        # B,C,H,W -> B,1,HW
        b = self.key_conv(x)
        b_max, l = torch.max(b, dim=1, keepdim=True)
        b_max = b_max.view(m_batchsize, -1, width*height)
        b_mean = torch.mean(b, dim=1, keepdim=True).view(m_batchsize, -1, width*height)
        # B,C,H,W -> B,HW,1
        c = self.query_conv(x)
        c_max, l = torch.max(c, dim=1, keepdim=True)
        c_max = c_max.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        c_mean = torch.mean(c, dim=1, keepdim=True).view(m_batchsize, -1, width*height).permute(0, 2, 1)

        # B,HW,HW
        s_max = torch.bmm(c_max, b_max)
        s_mean = torch.bmm(c_mean, b_mean)
        s = self.softmax(s_max.add(s_mean))

        # B,1,HW
        k_max = torch.bmm(a_max, s)
        k_mean = torch.bmm(a_mean, s)
        k = self.alpha_1 * k_max + self.beta_1 * k_mean
        # B,1,H,W
        k = k.view(m_batchsize, -1, height, width)

        # C,B,1,H,W
        out = torch.stack([k for i in range(C)], dim=0)
        # C,B,H,W
        out = out.squeeze(dim=2)
        # B,C,H,W
        print(np.size(out))
        out = out.permute(1, 0, 2, 3)

        return out


class CSPFM(Module):
    def __init__(self):
        super(CSPFM, self).__init__()

        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.mean = nn.AdaptiveAvgPool2d((1, 1))

        self.alpha_2 = Parameter(torch.zeros(1))
        self.beta_2 = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        # B,1,C
        d = self.max(x).view(m_batchsize, -1, 1).permute(0, 2, 1)
        e = self.mean(x).view(m_batchsize, -1, 1).permute(0, 2, 1)
        # B,C,1
        d_t = d.permute(0, 2, 1)
        e_t = e.permute(0, 2, 1)
        # B,C,C
        y_max = torch.bmm(d_t, d)
        y_mean = torch.bmm(e_t, e)
        y = self.softmax(y_max.add(y_mean))
        # B,1,C

        f_max = torch.bmm(d, y)
        f_mean = torch.bmm(e, y)
        f = self.alpha_2 * f_max + self.beta_2 * f_mean
        # B,C,1
        out = f.permute(0, 2, 1)
        # B,C,1,HW
        out = torch.stack([out for i in range(height * width)], dim=3)
        # B,C,HW
        out = torch.squeeze(out)
        # B,C,H,W
        out = out.view(m_batchsize, C, height, width)

        return out


class SCGSFM(Module):
    def __init__(self, in_dim, in_H, in_W):
        super(SCGSFM, self).__init__()

        self.N = in_H * in_W
        self.conv1 = Conv2d(in_channels=in_dim, out_channels=self.N, kernel_size=1)
        self.conv2 = Conv2d(in_channels=self.N, out_channels=in_dim, kernel_size=1)

        self.query_conv = Conv2d(in_channels=self.N, out_channels=self.N, kernel_size=1)
        self.value_conv = Conv2d(in_channels=self.N, out_channels=self.N, kernel_size=1)

        self.alpha_3 = Parameter(torch.zeros(1))
        self.beta_3 = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        N = self.N

        # B,N,H,W
        x = self.conv1(x)

        # B,N,HW
        r = self.value_conv(x).view(m_batchsize, N, height * width)
        g = self.query_conv(x).view(m_batchsize, N, height * width)
        # B,HW,N
        m = x.view(m_batchsize, N, height * width).permute(0, 2, 1)
        t = m

        # B,N,HW
        z_s = self.softmax(torch.bmm(m.permute(0, 2, 1), g))
        # B,HW,N
        z_c = self.softmax(torch.bmm(g.permute(0, 2, 1), m))

        # B,N,HW
        p_s = torch.bmm(z_s, r)
        # B,HW,N
        p_c = torch.bmm(z_c, t)

        # B,N,HW
        p_s = p_s.view(m_batchsize, -1, height, width)
        p_c = p_c.permute(0, 2, 1).view(m_batchsize, -1, height, width)

        p = self.alpha_3 * p_s + self.beta_3 * p_c
        out = self.conv2(p)

        return out

