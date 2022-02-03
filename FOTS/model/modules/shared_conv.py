import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from ...base.base_model import BaseModel

SPEEDUP_SCALE = 512


class SharedConv(BaseModel):
    '''
    sharded convolutional layers
    '''

    def __init__(self, bbNet: nn.Module, config):
        super(SharedConv, self).__init__(config)
        self.backbone = bbNet
        self.backbone.eval()
        # backbone as feature extractor
        """ for param in self.backbone.parameters():
            param.requires_grad = False """

        # Feature-merging branch
        # self.toplayer = nn.Conv2d(2048, 256, kernel_size = 1, stride = 1, padding = 0)  # Reduce channels

        # ## ResNet-50
        # self.mergeLayers0 = DummyLayer()                                 ## B x 2048 x H / 32 x W / 32
        # self.mergeLayers1 = HLayer(2048 + 1024, 128)                     ## B x 128 x H / 32 x W / 32
        # self.mergeLayers2 = HLayer(128 + 512, 64)                        ## B x 64 x H / 32 x W / 32
        # self.mergeLayers3 = HLayer(64 + 256, 32)                         ## B x 32 x H / 32 x W / 32
        # self.mergeLayers4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  ## B x 32 x H / 32 x W / 32
        # self.bn5 = nn.BatchNorm2d(32, momentum=0.003)

        # # ResNet-34
        # self.mergeLayers0 = DummyLayer()                                   ## B x 512 x H / 32 x W / 32
        # self.mergeLayers1 = HLayer(512 + 256, 128)                         ## B x 128 x H / 32 x W / 32
        # self.mergeLayers2 = HLayer(128 + 128, 64)                          ## B x 64 x H / 32 x W / 32
        # self.mergeLayers3 = HLayer(64 + 64, 32)                            ## B x 32 x H / 32 x W / 32
        # self.mergeLayers4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)    ## B x 128 x H / 32 x W / 32
        # self.bn5 = nn.BatchNorm2d(32, momentum=0.003)

        # ResNet-18
        self.mergeLayers0 = DummyLayer()                                   ## B x 512 x H / 32 x W / 32
        self.mergeLayers1 = HLayer(512 + 256, 128)                         ## B x 128 x H / 32 x W / 32
        self.mergeLayers2 = HLayer(128 + 128, 64)                          ## B x 64 x H / 32 x W / 32
        self.mergeLayers3 = HLayer(64 + 64, 32)                            ## B x 32 x H / 32 x W / 32
        self.mergeLayers4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)    ## B x 128 x H / 32 x W / 32
        self.bn5 = nn.BatchNorm2d(32, momentum=0.003)

        # Output Layer
        self.textScale = 512


    def forward(self, input):

        input = self.__mean_image_subtraction(input)

        # bottom up

        f = self.__forward_backbone(input)

        g = [None] * 4
        h = [None] * 4

        # i = 1
        h[0] = self.mergeLayers0(f[0])   ## H / 32 x W / 32
        # print('h[0]', h[0].shape)
        g[0] = self.__unpool(h[0])       ## H / 16 x W / 16
        # print('g[0]', g[0].shape)

        # i = 2
        h[1] = self.mergeLayers1(g[0], f[1])  ## H / 16 x W / 16
        # print('h[1]', h[1].shape)
        g[1] = self.__unpool(h[1])            ## H / 8 x W / 8
        # print('g[1]', g[1].shape)

        # i = 3
        h[2] = self.mergeLayers2(g[1], f[2])  ## H / 4 x W / 4
        # print('h[2]', h[2].shape)
        g[2] = self.__unpool(h[2])            ## H / 4 x W / 4
        # print('g[2]', g[2].shape)

        # i = 4
        h[3] = self.mergeLayers3(g[2], f[3])  ## H / 4 x W / 4
        # print('h[3]', h[3].shape)
        g[3] = self.__unpool(h[3])
        # print('g[3]', g[3].shape)

        # final stage
        final = self.mergeLayers4(h[3])       ## H / 4 x W / 4
        # print('final', final.shape)
        final = self.bn5(final)
        final = F.relu(final)

        return final

    def __forward_backbone(self, input):
        conv2 = None
        conv3 = None
        conv4 = None
        output = None # n * 7 * 7 * 2048

        for name, layer in self.backbone.named_children():
            input = layer(input)
            if name == 'layer1':
                conv2 = input
            elif name == 'layer2':
                conv3 = input
            elif name == 'layer3':
                conv4 = input
            elif name == 'layer4':
                output = input
                break

        return output, conv4, conv3, conv2

    def __unpool(self, input):
        _, _, H, W = input.shape
        return F.interpolate(input, mode='bilinear', scale_factor=2, align_corners=True)

    def __mean_image_subtraction(self, images, means = [123.68, 116.78, 103.94]):
        '''
        image normalization
        :param images: bs * w * h * channel
        :param means:
        :return:
        '''
        num_channels = images.data.shape[1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')
        for i in range(num_channels):
            images.data[:, i, :, :] -= means[i]

        return images


class DummyLayer(nn.Module):

    def forward(self, input_f):
        return input_f


class HLayer(nn.Module):

    def __init__(self, inputChannels, outputChannels):
        """

        :param inputChannels: channels of g+f
        :param outputChannels:
        """
        super(HLayer, self).__init__()

        self.conv2dOne = nn.Conv2d(inputChannels, outputChannels, kernel_size=1)
        self.bnOne = nn.BatchNorm2d(outputChannels, momentum=0.003)

        self.conv2dTwo = nn.Conv2d(outputChannels, outputChannels, kernel_size=3, padding=1)
        self.bnTwo = nn.BatchNorm2d(outputChannels, momentum=0.003)

    def forward(self, inputPrevG, inputF):
        input = torch.cat([inputPrevG, inputF], dim=1)
        output = self.conv2dOne(input)
        output = self.bnOne(output)
        output = F.relu(output)

        output = self.conv2dTwo(output)
        output = self.bnTwo(output)
        output = F.relu(output)

        return output