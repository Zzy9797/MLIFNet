from turtle import forward
from tables import split_type
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups,bias=False)


def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
    return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class LocalFeatureExtractor(nn.Module):

    def __init__(self, inplanes, planes, index):
        super(LocalFeatureExtractor, self).__init__()
        self.index = index

        norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()

        self.conv1_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn1_1 = norm_layer(planes)
        self.conv1_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = norm_layer(planes)

        self.conv2_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = norm_layer(planes)
        self.conv2_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = norm_layer(planes)

        self.conv3_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn3_1 = norm_layer(planes)
        self.conv3_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = norm_layer(planes)

        self.conv4_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn4_1 = norm_layer(planes)
        self.conv4_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = norm_layer(planes)
        
    def forward(self, x):

        patch_11 = x[:, :, 0:28, 0:28]
        patch_21 = x[:, :, 28:56, 0:28]
        patch_12 = x[:, :, 0:28, 28:56]
        patch_22 = x[:, :, 28:56, 28:56]

        out_1 = self.conv1_1(patch_11)
        out_1 = self.bn1_1(out_1)
        out_1 = self.relu(out_1)
        out_1 = self.conv1_2(out_1)
        out_1 = self.bn1_2(out_1)
        out_1 = self.relu(out_1)

        out_2 = self.conv2_1(patch_21)
        out_2 = self.bn2_1(out_2)
        out_2 = self.relu(out_2)
        out_2 = self.conv2_2(out_2)
        out_2 = self.bn2_2(out_2)
        out_2 = self.relu(out_2)

        out_3 = self.conv3_1(patch_12)
        out_3 = self.bn3_1(out_3)
        out_3 = self.relu(out_3)
        out_3 = self.conv3_2(out_3)
        out_3 = self.bn3_2(out_3)
        out_3 = self.relu(out_3)

        out_4 = self.conv4_1(patch_22)
        out_4 = self.bn4_1(out_4)
        out_4 = self.relu(out_4)
        out_4 = self.conv4_2(out_4)
        out_4 = self.bn4_2(out_4)
        out_4 = self.relu(out_4)

        out1 = torch.cat([out_1, out_2], dim=2)
        out2 = torch.cat([out_3, out_4], dim=2)
        out = torch.cat([out1, out2], dim=3)

        return out

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True))

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class channel_attention(nn.Module):
    def __init__(self,inchannel,split_rate=0.25):
        super(channel_attention,self).__init__()
        self.split_rate=split_rate
        self.conv=nn.Sequential(
            depthwise_conv(inchannel*2,inchannel*2,kernel_size=5,padding=2),
            conv1x1(inchannel*2,inchannel)
        )
        self.bn1=nn.BatchNorm2d(inchannel)
        self.bn2=nn.BatchNorm2d(inchannel)
        self.bn3=nn.BatchNorm2d(inchannel)
        self.bn4=nn.BatchNorm2d(inchannel)
        self.act=nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((3,3))
        self.fc=nn.Linear(inchannel*3*3,inchannel)

    def forward(self,x):
        channel_num=x.size()[1]
        batchsize=x.size()[0]
        split_channel=int(channel_num*self.split_rate)
        channel1, channel2,channel3,channel4 = torch.split(x, (split_channel,split_channel,split_channel,split_channel), dim=1)
        channel=channel1+channel2+channel3+channel4
        weight_1=self.act(self.bn1(self.conv(torch.cat([channel,channel1],dim=1))))
        weight_2=self.act(self.bn2(self.conv(torch.cat([channel,channel2],dim=1))))
        weight_3=self.act(self.bn3(self.conv(torch.cat([channel,channel3],dim=1))))
        weight_4=self.act(self.bn4(self.conv(torch.cat([channel,channel4],dim=1))))
        weight1=self.fc((self.avgpool(weight_1)+self.avgpool(channel1)).view(batchsize,-1)).unsqueeze(-1).unsqueeze(-1)
        weight2=self.fc((self.avgpool(weight_2)+self.avgpool(channel2)).view(batchsize,-1)).unsqueeze(-1).unsqueeze(-1)
        weight3=self.fc((self.avgpool(weight_3)+self.avgpool(channel3)).view(batchsize,-1)).unsqueeze(-1).unsqueeze(-1)
        weight4=self.fc((self.avgpool(weight_4)+self.avgpool(channel4)).view(batchsize,-1)).unsqueeze(-1).unsqueeze(-1)
        channel1=channel1*weight1
        channel2=channel2*weight2
        channel3=channel3*weight3
        channel4=channel4*weight4
        out=torch.cat([channel1,channel2,channel3,channel4],dim=1)
        return out



class MLIFB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(MLIFB, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv3x3(in_channels, in_channels,groups=4)
        self.c21 = nn.Conv2d(self.remaining_channels, self.remaining_channels,kernel_size=5,stride=1,padding=2,dilation=1,groups=self.remaining_channels)
        self.c22 = conv1x1(self.remaining_channels, in_channels,groups=4)
        self.c31 = nn.Conv2d(self.remaining_channels, self.remaining_channels,kernel_size=5,stride=1,padding=2,dilation=1,groups=self.remaining_channels)
        self.c32 = conv1x1(self.remaining_channels, in_channels,groups=4)
        self.c4 = conv1x1(self.remaining_channels, self.distilled_channels)
        self.act = nn.ReLU()
        self.attention = channel_attention(self.distilled_channels)
    def forward(self, input):
        out_c1 = self.c1(input)+input
        out_c1 = channel_shuffle(out_c1,4)
        distilled_c1, remaining_c1 = torch.split(self.act(out_c1), (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.c22(self.c21(remaining_c1))+out_c1
        out_c2 = channel_shuffle(out_c2,4)
        distilled_c2, remaining_c2 = torch.split(self.act(out_c2), (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.c32(self.c31(remaining_c2))+out_c2
        out_c3 = channel_shuffle(out_c3,4)
        distilled_c3, remaining_c3 = torch.split(self.act(out_c3), (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)
        out_c4 = self.act(out_c4)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.attention(out) + input
        return out_fused


class MLIFNet(nn.Module):

    def __init__(self, stages_repeats, stages_out_channels, num_classes=7):
        super(MLIFNet, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True),)
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        self.local = LocalFeatureExtractor(24, 96, 1)
        self.imdtb=MLIFB(96)
        output_channels = self._stage_out_channels[-1]

        self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True),)

        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):     
        x = self.conv1(x)        
        x1 = self.maxpool(x)      
        x2 = self.imdtb(self.stage2(x1)) + self.local(x1)
        x3 = self.stage3(x2)      
        x4 = self.stage4(x3)   
        x5 = self.conv5(x4)    
        x = x5.mean([2, 3])    
        x = self.fc(x)     
        return x

def mlifnet():
    model = MLIFNet([4, 8, 4], [24, 96, 192, 384, 1024])
    return model

