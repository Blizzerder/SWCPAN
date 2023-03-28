import torch
from torch import nn

from .common.modules import conv3x3, SwinModule,PatchMerging2
from .base_model import Base_model
from .builder import MODELS


class CrossSwinTransformer(nn.Module):
    def __init__(self, cfg, logger, n_feats=64, n_heads=4, head_dim=16, win_size=4,
                 n_blocks=3, cross_module=['pan', 'ms'], cat_feat=['pan', 'ms'], sa_fusion=False):
        super().__init__()
        self.cfg = cfg
        self.n_blocks = n_blocks
        self.cross_module = cross_module
        self.cat_feat = cat_feat
        self.sa_fusion = sa_fusion

        self.pan_conv_first = nn.Conv2d(1,n_feats,3,1,1)
        self.pan_merge = PatchMerging2(n_feats,n_feats,4)
        self.ms_conv_first = nn.Conv2d(cfg.ms_chans,n_feats,3,1,1)
        self.conv_last = nn.Sequential(nn.Conv2d(n_feats, n_feats // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(n_feats // 4, n_feats // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(n_feats // 4, n_feats, 3, 1, 1))
        pan_encoder = [
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
                       #1,64,2,2,4,16,4,True,False
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
        ]
        ms_encoder = [
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),          
        ]

        
        self.HR_tail = nn.Sequential(
            conv3x3(n_feats * len(cat_feat), n_feats * 4),
            nn.PixelShuffle(2), nn.ReLU(True), conv3x3(n_feats, n_feats * 4),
            nn.PixelShuffle(2), nn.ReLU(True), conv3x3(n_feats, n_feats),
            nn.ReLU(True), conv3x3(n_feats, cfg.ms_chans))

        self.pan_encoder = nn.Sequential(*pan_encoder)
        self.ms_encoder = nn.Sequential(*ms_encoder)

    def forward(self, pan, ms):
        pan_feat = self.pan_conv_first(pan)
        pan_feat = self.pan_encoder(pan_feat)
        #pan_feat =self.conv_last(pan_feat)
        ms_feat = self.ms_conv_first(ms)
        ms_feat = self.ms_encoder(ms_feat) + ms_feat
        #ms_feat = self.conv_last(ms_feat)

        cat_list = []
        if 'pan' in self.cat_feat:
            cat_list.append(pan_feat)
        if 'ms' in self.cat_feat:
            cat_list.append(ms_feat)

        output = self.HR_tail(torch.cat(cat_list, dim=1))

        if self.cfg.norm_input:
            output = torch.clamp(output, 0, 1)
        else:
            output = torch.clamp(output, 0, 2 ** self.cfg.bit_depth - .5)

        return output


@MODELS.register_module()
class SwinPan(Base_model):
    def __init__(self, cfg, logger, train_data_loader, test_data_loader1):
        super().__init__(cfg, logger, train_data_loader, test_data_loader1)

        model_cfg = cfg.get('model_cfg', dict())
        G_cfg = model_cfg.get('core_module', dict())

        self.add_module('core_module', CrossSwinTransformer(cfg=cfg, logger=logger, **G_cfg))

    def get_model_output(self, input_batch):
        input_pan = input_batch['input_pan']
        input_lr = input_batch['input_lr']
        output = self.module_dict['core_module'](input_pan, input_lr)
        return output

    def train_iter(self, iter_id, input_batch, log_freq=10):
        G = self.module_dict['core_module']
        G_optim = self.optim_dict['core_module']

        input_pan = input_batch['input_pan']
        input_lr = input_batch['input_lr']

        output = G(input_pan, input_lr)

        loss_g = 0
        loss_res = dict()
        loss_cfg = self.cfg.get('loss_cfg', {})
        if 'rec_loss' in self.loss_module:
            target = input_batch['target']
            rec_loss = self.loss_module['rec_loss'](
                out=output, gt=target
            )
            loss_g = loss_g + rec_loss * loss_cfg['rec_loss'].w
            loss_res['rec_loss'] = rec_loss.item()

        loss_res['full_loss'] = loss_g.item()

        G_optim.zero_grad()
        loss_g.backward()
        G_optim.step()

        self.print_train_log(iter_id, loss_res, log_freq)
