import torch
import torch.nn as nn
from torch.nn import functional as F


    

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


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        bn = nn.BatchNorm2d(out_channels)
        gelu = nn.GELU()
        layers = [conv1, bn, gelu]
        self.layers = nn.Sequential(*layers)
    
    def forward(self):
        out = self.layers()
        return out
        

class Changer(nn.Module):
    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(**kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)
        

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels // 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        
        self.neck_layer = FDAF(in_channels=self.channels // 2)
        
        # projection head
        self.discriminator = MixFFN(
            embed_dims=self.channels,
            feedforward_channels=self.channels,
            ffn_drop=0.,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            act_cfg=dict(type='GELU'))
                
    def base_forward(self, inputs):
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        
        return out

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        inputs1 = []
        inputs2 = []
        for input in inputs:
            f1, f2 = torch.chunk(input, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)
        
        out1 = self.base_forward(inputs1)
        out2 = self.base_forward(inputs2)
        out = self.neck_layer(out1, out2, 'concat')

        out = self.discriminator(out)
        out = self.cls_seg(out)

        return out



if '__name__' == '__main__':
    print()
    