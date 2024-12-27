import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import utils

from math import exp

class ConditionNet(nn.Module):
    def __init__(self, channels = 8):
        super(ConditionNet,self).__init__()
        self.convpre = nn.Conv2d(channels, channels, 3, 1, 1)
        self. conv1 = DenseBlock(channels, channels)
        self.down1 = nn.Conv2d(channels, 2*channels, stride=2, kernel_size=2)
        self.conv2 = DenseBlock(2*channels, 2*channels)
        self.down2 = nn.Conv2d(2*channels, 4*channels, stride=2, kernel_size=2)
        self.conv3 = DenseBlock(4*channels, 4*channels)

        self.Global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * channels, 4 * channels, 1, 1, 0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(4 * channels, 4 * channels, 1, 1, 0))
        self.context_g = DenseBlock(8 * channels, 4 * channels)

        self.context2 = DenseBlock(2 * channels, 2 * channels)
        self.context1 = DenseBlock(channels, channels)

        self.merge2 = nn.Sequential(nn.Conv2d(6*channels,2*channels,1,1,0),CALayer(2*channels,4),nn.Conv2d(2*channels,2*channels,3,1,1))
        self.merge1 = nn.Sequential(nn.Conv2d(3*channels,channels,1,1,0),CALayer(channels,4),nn.Conv2d(channels,channels,3,1,1))

        self.conv_last = nn.Conv2d(channels,3,3,1,1)


    def forward(self, x, mask):
        xpre = x/(torch.mean(x,1).unsqueeze(1)+1e-8)
        mask = torch.cat([mask,mask],1)
        x1 = self.conv1(self.convpre(torch.cat([xpre,x,mask],1)))
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv3(self.down2(x2))

        x_global = self.Global(x3)
        _,_,h,w = x3.size()
        x_global = x_global.repeat(1,1,h,w)
        x3 = self.context_g(torch.cat([x_global,x3],1))

        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear')
        x2 = self.context2(self.merge2(torch.cat([x2, x3], 1)))

        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x1 = self.context1(self.merge1(torch.cat([x1, x2], 1)))
        xout = self.conv_last(x1)

        return xout

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, gc)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)

        # initialize_weights(self.conv5, 0)
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out

class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x

def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class CoupleLayer(nn.Module):
    def __init__(self, channels, substructor, condition_length,  clamp=5.):
        super().__init__()

        channels = channels
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.conditional = True
        self.shadowpre = nn.Sequential(
            nn.Conv2d(4, channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.2))
        self.shadowpro = ShadowProcess(channels // 2)

        self.s1 = substructor(self.split_len1 + condition_length, self.split_len2*2)
        self.s2 = substructor(self.split_len2 + condition_length, self.split_len1*2)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp)) # 0.636*torch.atan ensures values between -1 and 1

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, c, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))
        c_star = self.shadowpre(c)
        c_star = self.shadowpro(c_star)

        if not rev:
            r2 = self.s2(x2, c_star)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(y1, c_star)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1

        else: # names of x and y are swapped!
            r1 = self.s1(x1, c_star)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(y2, c_star)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)

        return torch.cat((y1, y2), 1)

    def output_dims(self, input_dims):
        return input_dims

class ShadowProcess(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.process = UNetConvBlock(channels, channels)
        self.Attention = nn.Sequential(
            nn.Conv2d(channels,channels,3,1,1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.process(x)
        xatt = self.Attention(x)

        return xatt

class MultiscaleDense(nn.Module):
    def __init__(self,channel_in, channel_out, init):
        super(MultiscaleDense, self).__init__()
        self.conv_mul = nn.Conv2d(channel_out//2,channel_out//2,3,1,1)
        self.conv_add = nn.Conv2d(channel_out//2, channel_out//2, 3, 1, 1)
        self.down1 = nn.Conv2d(channel_out//2,channel_out//2,stride=2,kernel_size=2,padding=0)
        self.down2 = nn.Conv2d(channel_out//2, channel_out//2, stride=2, kernel_size=2, padding=0)
        self.op1 = DenseBlock(channel_in, channel_out, init)
        self.op2 = DenseBlock(channel_in, channel_out, init)
        self.op3 = DenseBlock(channel_in, channel_out, init)
        self.fuse = nn.Conv2d(3 * channel_out, channel_out, 1, 1, 0)

    def forward(self, x, s):
        s_mul = self.conv_mul(s)
        s_add = self.conv_add(s)

        x1 = x
        x2,s_mul2,s_add2 = self.down1(x),\
                           F.interpolate(s_mul, scale_factor=0.5, mode='bilinear'),F.interpolate(s_add, scale_factor=0.5, mode='bilinear')
        x3, s_mul3, s_add3 = self.down2(x2), \
                             F.interpolate(s_mul, scale_factor=0.25, mode='bilinear'), F.interpolate(s_add,scale_factor=0.25,mode='bilinear')
        x1 = self.op1(torch.cat([x1,s_mul*x1+s_add],1))
        x2 = self.op2(torch.cat([x2,s_mul2*x2+s_add2],1))
        x3 = self.op3(torch.cat([x3,s_mul3*x3+s_add3],1))
        x2 = F.interpolate(x2, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x3 = F.interpolate(x3, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x = self.fuse(torch.cat([x1, x2, x3], 1))

        return x

def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return MultiscaleDense(channel_in, channel_out, init)
            else:
                return MultiscaleDense(channel_in, channel_out, init)
            # return UNetBlock(channel_in, channel_out)
        else:
            return None

    return constructor

class InvISPNet(nn.Module):
    def __init__(self, color_model_pretrained_file, channel_in=3, subnet_constructor=subnet('DBNet'), block_num=4):
        super().__init__()
        operations = []
        self.condition = ConditionNet()
        utils.load_model(color_model_pretrained_file, self.condition)
        #self.condition.load_state_dict(torch.load(color_model_pretrained_file))
        for p in self.parameters():
            p.requires_grad = False

        channel_num = 16  # total channels at input stage
        self.CG0 = nn.Conv2d(channel_in, channel_num, 1, 1, 0)
        self.CG1 = nn.Conv2d(channel_num, channel_in, 1, 1, 0)
        self.CG2 = nn.Conv2d(channel_in, channel_num, 1, 1, 0)
        self.CG3 = nn.Conv2d(channel_num, channel_in, 1, 1, 0)

        for j in range(block_num):
            b = CoupleLayer(channel_num, substructor = subnet_constructor, condition_length=channel_num//2)  # one block is one flow step.
            operations.append(b)

        self.operations = nn.ModuleList(operations)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

    def forward(self, input, mask, gt, rev=False):
        b, c, m, n = input.shape
        maskcolor = self.condition(input,mask)
        maskfea = torch.cat([maskcolor,mask],1)

        if not rev:
            x = input
            out = self.CG0(x)
            out_list = []
            for op in self.operations:
                out_list.append(out)
                out = op.forward(out, maskfea, rev)
            out = self.CG1(out)
        else:
            out = self.CG2(gt)
            out_list = []
            for op in reversed(self.operations):
                out = op.forward(out, maskfea, rev)
                out_list.append(out)
            out_list.reverse()
            out = self.CG3(out)
        
        return out, maskcolor
