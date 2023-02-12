import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, Callable, List, Optional, Type, Union
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode

    

def fusion(x1, x2, policy):
    """Specify the form of feature fusion"""
    
    _fusion_policies = ['concat', 'sum', 'diff', 'abs_diff']
    assert policy in _fusion_policies, 'The fusion policies {} are ' \
        'supported'.format(_fusion_policies)
    
    if policy == 'concat':
        x = torch.cat([x1, x2], dim=1)
    elif policy == 'sum':
        x = x1 + x2
    elif policy == 'diff':
        x = x2 - x1
    elif policy == 'Lp_distance':
        x = torch.abs(x1 - x2)

    return x


class FDAF(nn.Module):
    """Flow Dual-Alignment Fusion Module.
    Args:
        in_channels (int): Input channels of features.
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
    """

    def __init__(self, in_channels):
        super().__init__()
        kernel_size = 5
        self.flow_make = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels*2, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True, groups=in_channels*2),
            nn.InstanceNorm2d(in_channels*2),
            nn.GELU(),
            nn.Conv2d(in_channels*2, 4, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x1, x2, fusion_policy=None):
        """Forward function."""

        output = torch.cat([x1, x2], dim=1)
        flow = self.flow_make(output)
        f1, f2 = torch.chunk(flow, 2, dim=1)
        x1_feat = self.warp(x1, f1) - x2
        x2_feat = self.warp(x2, f2) - x1
        
        if fusion_policy == None:
            return x1_feat, x2_feat
        
        output = fusion(x1_feat, x2_feat, fusion_policy)
        return output

    @staticmethod
    def warp(x, flow):
        n, c, h, w = x.size()

        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(x, grid, align_corners=True)
        return output


class MixFFN(nn.Module):
    """
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
    """

    def __init__(self,
                 embed_dims=4,
                 feedforward_channels=4,
                 ffn_drop=0.2):
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.activate = nn.GELU()

        in_channels = embed_dims
        fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=2 // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = nn.Sequential(*layers)


    def forward(self, x, identity=None):
        out = self.layers(x)
        if identity is None:
            identity = x
        return identity + out




class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        bn = nn.BatchNorm2d(out_channels)
        gelu = nn.GELU()
        layers = [conv1, bn, gelu]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, out):
        out = self.layers(out)
        return out
        

class Changer(nn.Module):
    def __init__(self, in_channels : List[int], out_channels : int ,channels : int, interpolate_mode='bilinear',**kwargs):
        super().__init__(**kwargs)
        
        self.input_transform = 'multiple_select' # multiple_select, resize_concat
        self.interpolate_mode = interpolate_mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        
        num_inputs = len(self.in_channels) # 4

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i]//2, # 128, 256, 512, 1024
                    out_channels=self.channels, # 1024
                    kernel_size=1,
                    stride=1))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,  # 1024 * 4
            out_channels=self.channels // 2, # 512
            kernel_size=1)
        
        self.FDAF = FDAF(in_channels=self.channels // 2)
        
        self.cls_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)
        
        # projection head
        self.MixFFN = MixFFN(
            embed_dims=self.channels, #1024
            feedforward_channels=self.channels, #1024
            ffn_drop=0.2,
            )
        
    def base_forward(self, inputs):
        outs = []
        for i in range(len(inputs)):
            conv = self.convs[i](inputs[i])
            outs.append(conv)
        out = torch.cat(outs, dim=1)
        out = self.fusion_conv(out)

        return out
                        
    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs) # 
        inputs1 = []
        inputs2 = []
        for input in inputs:
            f1, f2 = torch.chunk(input, 2, dim=1)
            inputs1.append(f1) #[64, 128, 256, 512]
            inputs2.append(f2) #[64, 128, 256, 512]
        out1 = self.base_forward(inputs1)
        out2 = self.base_forward(inputs2)
        
        out = self.FDAF(out1, out2, 'concat')
        out = self.MixFFN(out)
        out = self.cls_seg(out)

        return out

    def _transform_inputs(self, inputs):
        feat_size = int(inputs[0].shape[2])
        
        
        if self.input_transform == 'resize_concat':
            
            upsampled_inputs = [
                Resize(
                    size=(feat_size, feat_size),
                    interpolation=InterpolationMode.BILINEAR)(x) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
            
        elif self.input_transform == 'multiple_select':
            upsampled_inputs = [
                Resize(
                    size=(feat_size, feat_size),
                    interpolation=InterpolationMode.BILINEAR,)(x) for x in inputs
            ]
            inputs = upsampled_inputs

        return inputs
    
    
    


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = Changer([128, 256, 512, 1024], 1, 1024).to(device)
    x1 = torch.randn(1, 128, 128, 128)
    x2 = torch.randn(1, 256, 64, 64)
    x3 = torch.randn(1, 512, 32, 32)
    x4 = torch.randn(1, 1024, 16, 16)
    input= x1, x2, x3, x4
    output = model(input)
    print(output.size())

    
    