import torch
import torch.nn as nn
from timm.models.layers.helpers import to_2tuple
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from KANLinear import KANLinear, KAN
from kan_convs.kan_conv import KANConv2DLayer


"""
配备多头自注意力 （MHSA） 的模型在计算机视觉方面取得了显着的性能。它们的计算复杂度与输入特征图中的二次像素数成正比，导致处理速度缓慢，尤其是在处理高分辨率图像时。
为了规避这个问题，提出了一种新型的代币混合器作为MHSA的替代方案：基于FFT的代币混合器涉及类似于MHSA的全局操作，但计算复杂度较低。
在这里，我们提出了一种名为动态过滤器的新型令牌混合器以缩小上述差距。
DynamicFilter 模块通过频域滤波和动态调整滤波器权重，能够对图像进行复杂的增强和处理。
"""

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    # mlp_ratio=4
    # drop=0.0
    # expansion_ratio=2
    # num_filters=4

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        # self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc1 = KANLinear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        # self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.fc2 = KANLinear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        device = next(self.parameters()).device  # 获取模型所在的设备
        x = x.to(device)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

#   num_filters: 4 / 2 / 1
class DynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=False,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize

        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        # self.pwconv1 = KANLinear(dim, self.med_channels)
        # self.pwconv1 = KANConv2DLayer(dim,self.med_channels, kernel_size=1, groups=1)

        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()

        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)
        # self.pwconv2 = KANLinear(self.med_channels, dim)
        # self.pwconv2 = KANConv2DLayer(self.med_channels, dim, kernel_size=1, groups=1)

    def forward(self, x):
        device = next(self.parameters()).device  # 获取模型所在的设备
        x = x.to(device)

        B, H, W, C = x.shape
        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters,
                                                          -1).softmax(dim=1)

        #  pwconv: squeeze dim
        #  pwconv: 1X1kanconv
        # x = x.permute(0, 3, 1, 2).contiguous()
        # x = self.pwconv1(x)
        # x = x.permute(0, 2, 3, 1).contiguous()

        #  pwconv: kanlinear
        # x = x.view(-1,C)
        # x = self.pwconv1(x)
        # x = x.view(B, H, W, -1)
        # x = self.act1(x)

        # pwconv: nnlinear
        x = self.pwconv1(x)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho').contiguous()

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1],
                                                    x.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)

        x = x * weight

        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho').contiguous()
        x = self.act2(x)

        #  pwconv: 1X1kanconv
        # x = x.permute(0, 3, 1, 2).contiguous()
        # x = self.pwconv2(x)
        # x = x.permute(0, 2, 3, 1).contiguous()

        #  pwconv: kanlinear
        # b1, h1, w1, c1 = x.shape
        # x = x.view(b1*h1*w1, c1)
        # x = self.pwconv2(x)
        # x = x.view(b1, h1, w1, -1)

        x = self.pwconv2(x)

        return x


if __name__ == '__main__':
    block = DynamicFilter(32, size=64) # size==H,W
    input = torch.rand(3, 64, 64, 32)
    output = block(input)
    print(input.size())
    print(output.size())