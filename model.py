import layers
import torch
import torch.nn as nn

class Unet(nn.Module):
    def __init__(self, nh, num_classes):
        super(Unet, self).__init__()
                
        self.dconv_down1 = self.double_conv(3, nh)
        self.dconv_down2 = self.double_conv(nh, nh)
        self.dconv_down3 = self.double_conv(nh, nh)
        self.dconv_down4 = self.double_conv(nh, nh)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = layers.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = self.double_conv(nh+nh, nh)
        self.dconv_up2 = self.double_conv(nh+nh, nh)
        self.dconv_up1 = self.double_conv(nh+nh, nh)
        
        self.conv_last = nn.Conv2d(nh, num_classes, 1)


    def double_conv(self, in_channels, out_channels, use_batchnorm=False):
        if use_batchnorm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )   
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )   
        

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        return out
