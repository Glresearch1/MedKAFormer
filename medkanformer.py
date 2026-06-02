import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
# from KANConv import KAN_Convolutional_Layer
# from kan_convs.kan_conv import KANConv2DLayer
from kan_convs.kan import KAN
from kan_convs.utils import L1
# from model_test import CBAM
from KANLinear import KANLinear, KAN
from modules.dynamicfilter import DynamicFilter as dynamicfilter
from modules.contranorm import ContraNorm
from modules.od_kaconv_all import KANConv2DLayer


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = KANLinear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = KANLinear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # print('mlp_x shape is',x.shape)
        return x


class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"
    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size=kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation*(kernel_size-1)//2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self,q,k,v):
        #B, C//3, H, W
        B,d,H,W = q.shape
        q = q.reshape([B, d//self.head_dim, self.head_dim, 1, H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]).permute(0, 1, 4, 2, 3)  #B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.1, kernel_size=3,
                 dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"

        self.qkv = KANConv2DLayer(dim, dim * 3, kernel_size=1, spline_order=3, groups=1, padding=0, stride=1,
                                  dilation=1, grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0)

        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])

        self.proj = KANLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4,
                                                                                                 5).contiguous()
        x = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2).contiguous()
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W, C // self.num_dilation
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C).contiguous()
        B, H, W, C = x.shape
        x = x.view(-1,C)
        x = self.proj(x)
        x = x.view(B, H, W, C)

        x = self.proj_drop(x)
        return x


class DilateBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2, 3],
                 cpe_per_block=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = ContraNorm(dim=dim, scale=0.1, dual_norm=True, pre_norm=True, temp=1.0, learnable=True,
                           positive=False, identity=False)

        self.attn = MultiDilatelocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = ContraNorm(dim=dim, scale=0.1, dual_norm=True, pre_norm=True, temp=1.0, learnable=True,
                                   positive=False, identity=False)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = KAN([dim, mlp_hidden_dim, dim])

    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b,h,w,c = x.shape
        x = x.view(b,-1,c)
        x = self.norm1(x.clone())
        x = x.view(b,h,w,c)
        x = x + self.drop_path(self.attn(x))

        b,h,w,c = x.shape
        x = x.view(b,-1,c)
        x = self.norm2(x)

        b,n,c = x.shape
        x1 = x.view(-1,c)
        x = x.view(b,h,w,c)
        x = x + (self.drop_path(self.mlp(x1))).view(b,h,w,c)

        x = x.permute(0, 3, 1, 2).contiguous()
        return x



class GlobalAttention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim,  num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = KANLinear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = KANLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(-1,C)
        qkv = self.qkv(x)

        qkv = qkv.reshape(B, H * W, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        B, H, W, C = x.shape
        x = x.view(-1,C)
        x = self.proj(x)
        x = x.view(B, H, W, C)
        x = self.proj_drop(x)
        return x


class GlobalBlock(nn.Module):
    """
    Implementation of Transformer
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cpe_per_block=False):
        super().__init__()
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = ContraNorm(dim=dim, scale=0.1, dual_norm=True, pre_norm=True, temp=1.0, learnable=True, positive=False, identity=False)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = ContraNorm(dim=dim, scale=0.1, dual_norm=True, pre_norm=True, temp=1.0, learnable=True, positive=False, identity=False)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = KAN([dim, mlp_hidden_dim, dim])



    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)

        x = x.permute(0, 2, 3, 1)

        b,h,w,c = x.shape
        x = x.view(b,-1,c)
        x1 = self.norm1(x)
        x = x.view(b,h,w,c)
        x1 = x1.view(b,h,w,c)

        b1,h1,w1,c1 = x1.shape
        if isinstance(w1, torch.Tensor):
            w1 = w1.item()
        block = dynamicfilter(dim=c1, size=w1).to(x1.device)

        x1 = block(x1)
        x1 = self.drop_path(x1)
        x = x + x1

        b,h,w,c = x.shape
        x1 = x.view(b,-1,c)
        x = x.view(b,h,w,c)
        x1 = self.norm2(x1)

        x1 = x1.view(-1, c)
        x = x + (self.drop_path(self.mlp(x1))).view(b,h,w,c)
        x = x.permute(0, 3, 1, 2)

        return x



class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, in_chans=3, hidden_dim=48,
                 patch_size=4, embed_dim=96, patch_way=None, l1_penalty=0.01):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.img_size = img_size
        self.dropout = nn.Dropout(0.2)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm1 = nn.BatchNorm2d(in_chans)


        assert patch_way in ['overlaping', 'nonoverlaping', 'pointconv'], "the patch embedding way isn't exist!"

        if patch_way == "nonoverlaping":
            self.proj = KANConv2DLayer(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size)

        elif patch_way == "overlaping":
            self.proj = nn.Sequential(
                KANConv2DLayer(in_chans, hidden_dim, kernel_size=3, stride=1, padding=1),  # 224x224
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                L1(KANConv2DLayer(hidden_dim, int(hidden_dim * 2), kernel_size=3, stride=2, padding=1), l1_penalty),
                # 112x112
                nn.BatchNorm2d(int(hidden_dim * 2)),
                nn.GELU(),
                L1(KANConv2DLayer(int(hidden_dim * 2), int(hidden_dim * 3), kernel_size=3, stride=1, padding=1),
                   l1_penalty),  # 112x112
                nn.BatchNorm2d(int(hidden_dim * 3)),
                nn.GELU(),
                L1(KANConv2DLayer(int(hidden_dim * 3), embed_dim, kernel_size=3, stride=2, padding=1), l1_penalty),
                # 56x56
            )

        else:
            self.proj = nn.Sequential(
                KANConv2DLayer(in_chans, hidden_dim, kernel_size=3, stride=2, padding=1),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Dropout(p=0.5),
                KANConv2DLayer(int(hidden_dim), int(embed_dim), kernel_size=3, stride=2, padding=1),  # 56x56
                nn.BatchNorm2d(int(embed_dim)),
                nn.GELU(),
            )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        x = self.norm2(x)

        return x




class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """
    def __init__(self, in_channels, out_channels, merging_way, cpe_per_satge, norm_layer=nn.BatchNorm2d):
        super().__init__()
        assert merging_way in ['conv3_2', 'conv2_2', 'avgpool3_2', 'avgpool2_2'], \
            "the merging way is not exist!"
        self.cpe_per_satge = cpe_per_satge

        if merging_way == 'conv3_2':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                norm_layer(out_channels),

            )
        elif merging_way == 'conv2_2':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                norm_layer(out_channels),
            )
        elif merging_way == 'avgpool3_2':
            self.proj = nn.Sequential(
                nn.AvgPool2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                norm_layer(out_channels),
            )
        else:
            self.proj = nn.Sequential(
                nn.AvgPool2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                norm_layer(out_channels),
            )
        if self.cpe_per_satge:
            self.pos_embed = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels)

    def forward(self, x):
        x = self.proj(x)
        if self.cpe_per_satge:
            x = x + self.pos_embed(x)
        return x


class Dilatestage(nn.Module):
    """ A basic Dilate Transformer layer for one stage.
    """
    def __init__(self, dim, depth, num_heads, kernel_size, dilation,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, cpe_per_satge=False, cpe_per_block=False,
                 downsample=True, merging_way=None):

        super().__init__()
        self.blocks = nn.ModuleList([
            DilateBlock(dim=dim, num_heads=num_heads,
                        kernel_size=kernel_size, dilation=dilation,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block)
            for i in range(depth)])

        self.downsample = PatchMerging(dim, int(dim * 2), merging_way, cpe_per_satge) if downsample else nn.Identity()
        self.dropout = nn.Dropout(0.3)


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        x = self.dropout(x)
        return x


class Globalstage(nn.Module):
    """ A basic Transformer layer for one stage."""
    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cpe_per_satge=False, cpe_per_block=False,
                 downsample=True, merging_way=None):

        super().__init__()
        self.blocks = nn.ModuleList([
            GlobalBlock(dim=dim, num_heads=num_heads,
                        mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block)
            for i in range(depth)])

        self.downsample = PatchMerging(dim, int(dim*2), merging_way, cpe_per_satge) if downsample else nn.Identity()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        x = self.dropout(x)
        return x


class MedKAFormer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], kernel_size=3, dilation=[1, 2, 3],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 merging_way='conv3_2',
                 patch_way='nonoverlaping',
                 dilate_attention=[True, True, False, False],
                 downsamples=[True, True, True, False],
                 cpe_per_satge=True, cpe_per_block=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim, patch_way=patch_way,l1_penalty=0.01)

        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.stages = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if dilate_attention[i_layer]:
                stage = Dilatestage(dim=int(embed_dim * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    kernel_size=kernel_size,
                                    dilation=dilation,
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=downsamples[i_layer],
                                    cpe_per_block=cpe_per_block,
                                    cpe_per_satge=cpe_per_satge,
                                    merging_way=merging_way
                                    )
            else:
                stage = Globalstage(dim=int(embed_dim * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=downsamples[i_layer],
                                    cpe_per_block=cpe_per_block,
                                    cpe_per_satge=cpe_per_satge,
                                    merging_way=merging_way
                                    )
            self.stages.append(stage)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = KANLinear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        self.Dropout = nn.Dropout(0.5)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def forward_features(self, x):
        x = self.patch_embed(x)

        for stage in self.stages:
            x = stage(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.Dropout(x)
        x = self.head(x)
        return x


@register_model
def medkaformer_tiny(pretrained=True, **kwargs):
    # model = MedKAFormer(depths=[2, 2, 6, 2], embed_dim=72, num_heads=[3, 6, 12, 24], **kwargs)
    # model = MedKAFormer(depths=[1, 1, 3, 1], embed_dim=48, num_heads=[3, 3, 6, 9], mlp_ratio=4, **kwargs)
    # model = MedKAFormer(depths=[1, 1, 3, 1], embed_dim=36, num_heads=[3, 3, 6, 6], mlp_ratio=4, **kwargs)
    model = MedKAFormer(
        depths=[1, 1, 1, 1],
        embed_dim=36,
        num_heads=[3, 3, 3, 3],
        mlp_ratio=4,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def medkaformer_small(pretrained=True, **kwargs):
    model = MedKAFormer(
        depths=[1, 1, 3, 1],
        embed_dim=36,
        num_heads=[3, 3, 6, 6],
        mlp_ratio=4,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def medkaformer_base(pretrained=True, **kwargs):
    model = MedKAFormer(depths=[4, 8, 10, 3], embed_dim=96, num_heads=[3, 6, 12, 24], **kwargs)
    model.default_cfg = _cfg()
    return model




def count_kanlinear_params(model):
    kanlinear_params = 0

    for name, module in model.named_modules():
        if isinstance(module, KANLinear):
            base_weight_params = module.base_weight.numel()
            spline_weight_params = module.spline_weight.numel()
            spline_scaler_params = module.spline_scaler.numel() if hasattr(module, 'spline_scaler') else 0
            total_kanlinear_params = base_weight_params + spline_weight_params + spline_scaler_params
            kanlinear_params += total_kanlinear_params

            print(f'KANLinear Layer: {name}')
            print(f'  base_weight params: {base_weight_params}')
            print(f'  spline_weight params: {spline_weight_params}')
            if spline_scaler_params:
                print(f'  spline_scaler params: {spline_scaler_params}')
            print(f'  Total KANLinear params: {total_kanlinear_params}')

    return kanlinear_params



def count_total_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params



def count_non_kanlinear_params(model, kanlinear_params):
    total_params = count_total_params(model)
    non_kanlinear_params = total_params - kanlinear_params
    return non_kanlinear_params



def compute_kanlinear_ratio(model):
    kanlinear_params = count_kanlinear_params(model)
    total_params = count_total_params(model)
    ratio = kanlinear_params / total_params
    return kanlinear_params, total_params, ratio


if __name__ == "__main__":
    x = torch.rand([2, 3, 224, 224])
    m = medkaformer_tiny(pretrained=False)
    y = m(x)

    total_params = count_total_params(m)
    total_params_million = total_params / 1e6
    print(f"Total parameters: {total_params_million:.2f}M")

    kanlinear_params, total_params, ratio = compute_kanlinear_ratio(m)
    kanlinear_params_million = kanlinear_params / 1e6
    print(f"KANLinear parameters: {kanlinear_params_million:.2f}M")
    print(f"Non-KANLinear parameters: {(total_params - kanlinear_params) / 1e6:.2f}M")
    print(f"KANLinear parameters ratio: {ratio:.4f}")
