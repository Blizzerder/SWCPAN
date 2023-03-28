import torch
import torch.nn as nn
from logging import Logger
from pytorch_wavelets import DWTForward, DWTInverse, DWT1DForward, DWT1DInverse  # or simply DWT1D, IDWT1D
from mmcv import Config
from model.base_net import *
from models.base_model import Base_model
from models.builder import MODELS
from models.common.utils import up_sample, down_sample, get_hp, get_lp, channel_pooling
from models.common.modules import ResBlock, ResChAttnBlock, Patch_Discriminator, conv3x3, build_norm_layer
import torch.nn.functional as F
from model.nonlocal_block import *
torch.autograd.set_detect_anomaly = True




def data_normal(orign_data):
    d_min = orign_data.min()
    if d_min < 0:
        orign_data = orign_data+torch.abs(d_min)
        d_min = orign_data.min()
    d_max = orign_data.max()
    dst = d_max -d_min
    norm_data = (orign_data - d_min).true_divide(dst)
    return norm_data


class att_spatial(nn.Module):
    def __init__(self):
        super(att_spatial, self). __init__()
        kernel_size = 7
        block = [
            ConvBlock(2, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(6):
            block.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        self.block = nn.Sequential(*block)
        self.spatial = ConvBlock(2, 1, 3, 1, 1, activation='prelu', norm=None, bias = False)
        
    def forward(self, x):
        x = self.block(x)
        x_compress = torch.cat([torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)], dim=1)
        x_out = self.spatial(x_compress)

        scale = F.sigmoid(x_out) # broadcasting
        return scale


class EmbNet(nn.Module):
    def __init__(self, logger: Logger, ms_chans, n_blocks=1, n_feats=32, norm_type='BN',
                 basic_block=ResBlock):
        super(EmbNet, self).__init__()

        self.net = []
        self.net.append(conv3x3(ms_chans, n_feats))
        # 32 x 256 x 256
        if norm_type is not None:
            self.net.append(build_norm_layer(logger, n_feats, norm_type))
        for i in range(n_blocks):
            self.net.append(nn.ReLU(True))
            self.net.append(basic_block(logger, n_feats, norm_type))
        # 32 x 256 x 256
        self.net.append(nn.ReLU(True))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        # 32 x 256 x 256
        return self.net(x)




class Generator(nn.Module):     #生成模型
    def __init__(self, num_channels, upscale = 4):     #num_channels波段数 base_filter基过滤器 upscale上采样倍数
        
        super(Generator, self).__init__()

        num_channels = 5
        out_channels = 4
        self.upscale = upscale
        self.head = ConvBlock(num_channels, 48, 9, 1, 4, activation='relu', norm=None, bias=True)
        self.body = ConvBlock(48, 32, 5, 1, 2, activation='relu', norm=None, bias=True)
        self.output_conv = ConvBlock(32, out_channels, 5, 1, 2, activation='relu', norm=None, bias=True)
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x_pan, b_ms, l_ms):
        # x_pan：PAN图像        256*256*1
        # b_ms：上采样的ms图像  256*256*4
        # l_ms：原ms图像 64*64*4
        
        '''
        NDWI = ((l_ms[:, 1, :, :] - l_ms[:, 3, :, :]) / (l_ms[:, 1, :, :] + l_ms[:, 3, :, :])).unsqueeze(1)
        NDWI = F.interpolate(NDWI, scale_factor=self.upscale, mode='bicubic')
        #NDWI归一化水体指数
        NDVI = ((l_ms[:, 3, :, :] - l_ms[:, 2, :, :]) / (l_ms[:, 3, :, :] + l_ms[:, 2, :, :])).unsqueeze(1)
        NDVI = F.interpolate(NDVI, scale_factor=self.upscale, mode='bicubic')
        #NDVI归一化植被指数
        '''
        x_f = torch.cat([b_ms, x_pan], 1)
        #拼接
        x_f = self.head(x_f)
        #第一次卷积
        x_f = self.body(x_f)
        #第二次卷积
        x_f = self.output_conv(x_f)
        output = x_f
        return output

@MODELS.register_module()
class UCGAN(Base_model):
    def __init__(self, cfg, logger, train_data_loader, test_data_loader1):
        super(UCGAN, self).__init__(cfg, logger, train_data_loader, test_data_loader1)  #转到Base_model
        ms_chans = cfg.get('ms_chans', 4)               #获取ms的波段数
        model_cfg = cfg.get('model_cfg', dict())        #获取model的配置信息
        G_cfg = model_cfg.get('core_module', dict())    #G_cfg是获取生成模型的配置信息
        #D_cfg = model_cfg.get('Discriminator', dict()) #D_cfg是获取判别器的配置信息 这里没有

        self.add_module('core_module', Generator(7,**G_cfg))   #先生成一个Generator 再调用add_module将模型信息赋过去
        #self.add_module('core_module', Generator(cfg=cfg, logger=logger, ms_chans=ms_chans, **G_cfg))       #添加灵魂
        #self.add_module('Discriminator', Patch_Discriminator(logger=logger, in_channels=ms_chans*2+1, **D_cfg))
        self.to_pan_mode = model_cfg.get('to_pan_mode', 'max')  #池化模式

    def get_model_output(self, input_batch):
        input_pan = input_batch['input_pan']
        input_lr = input_batch['input_lr']
        input_lr_u = up_sample(input_lr)
        output = self.module_dict['core_module'](input_pan, input_lr_u, input_lr)
        return output

    def train_iter(self, iter_id, input_batch, log_freq=10):
        G = self.module_dict['core_module']         #生成器
        #D = self.module_dict['Discriminator']
        G_optim = self.optim_dict['core_module']    #设置参数
        #D_optim = self.optim_dict['Discriminator']

        input_pan = input_batch['input_pan']        #输入的pan图像         1*256*256
        input_pan_l = input_batch['input_pan_l']    #高斯下采样的pan图像    1*64*64
        input_lr = input_batch['input_lr']          #输入的ms图像          4*64*64            
        input_lr_u = up_sample(input_lr)            #输入的ms图像进行上采样 4*256*256
        target = input_batch['target']
        # x_pan：PAN图像        256*256*1
        # b_ms：上采样的ms图像  256*256*4
        # l_ms：原ms图像        64*64*4
        output = G(input_pan, input_lr_u, input_lr)     #4*256*256
        #print(output)
        fake_lr_u = up_sample(down_sample(output))      #先下采样再上采样4*256*256
        fake_pan = channel_pooling(output, mode=self.to_pan_mode)   #对output进行最大池化得到单通道图像  
        #output_cyc = G(input_pan_rept, fake_lr_u)
        
        loss_g = 0
        loss_res = dict()
        loss_cfg = self.cfg.get('loss_cfg', {})



        if 'all_rec_loss' in self.loss_module:
            all_rec_loss = self.loss_module['all_rec_loss'](out=data_normal(output),gt=data_normal(target))
            loss_g = loss_g + all_rec_loss
            loss_res['all_rec_loss'] = all_rec_loss.item()
        loss_res['full_loss'] = loss_g.item()

        G_optim.zero_grad()
        loss_g.backward()
        G_optim.step()

        self.print_train_log(iter_id, loss_res, log_freq)

