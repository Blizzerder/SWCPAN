import torch.nn as nn

from .swin_transformer import PatchEmbed, PatchMerging, BasicLayer


class Down(nn.Module):
    def __init__(self, down_scale=2, in_dim=64, depths=(2, 2, 6, 2),chans = 4,size = 256 , downscale = 1):
        super(Down, self).__init__()
        #Patch拆分模块
        self.inc = PatchEmbed(img_size=size, patch_size=down_scale, in_chans=chans, embed_dim=in_dim,
                              norm_layer=nn.LayerNorm)
        # 第一个Swin特征提取块
        self.down1 = DownBlock(in_channels=in_dim, out_channels=in_dim * 2, resolution=size // down_scale,
                               downsample=PatchMerging, cur_depth=depths[0])
        # 第二个Swin特征提取块
        self.down2 = DownBlock(in_channels=in_dim * 2, out_channels=in_dim * 4, resolution=size // down_scale // down_scale,
                               downsample=PatchMerging, cur_depth=depths[1])
        # 第三个Swin特征提取块
        self.down3 = DownBlock(in_channels=in_dim * 4, out_channels=in_dim * 8, resolution=size // down_scale // down_scale // down_scale,
                               downsample=PatchMerging, cur_depth=depths[2])
        # 第四个Swin特征提取块
        self.down4 = DownBlock(in_channels=in_dim * 8, out_channels=in_dim * 8, resolution=size // down_scale // down_scale // down_scale // down_scale,
                               downsample=PatchMerging, cur_depth=depths[3])

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #返回五个阶段的输出结果
        return x1, x2, x3, x4, x5
        #return x5

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, downsample, cur_depth):
        super(DownBlock, self).__init__()
        self.layer = BasicLayer(dim=in_channels,
                                input_resolution=(resolution, resolution),
                                depth=cur_depth,
                                num_heads=in_channels // 32,
                                window_size=8,
                                mlp_ratio=1,
                                qkv_bias=True, qk_scale=None,
                                drop=0., attn_drop=0.,
                                drop_path=0.,
                                norm_layer=nn.LayerNorm)

        if downsample is not None:
            self.downsample = downsample((resolution, resolution), in_channels, out_channels)
        else:
            self.downsample = None

    def forward(self, x):
        x_o = self.layer(x)

        if self.downsample is not None:
            x_o = self.downsample(x_o)

        return x_o

