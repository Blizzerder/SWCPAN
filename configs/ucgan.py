# ---> GENERAL CONFIG <---
name = 'ucgan_GF-2'
description = 'test ucgan on WV-3/GF-2 dataset'

model_type = 'UCGAN'
work_dir = f'data/PSData3/model_out/{name}'
log_dir = f'logs/{model_type.lower()}'
log_file = f'{log_dir}/{name}.log'
log_level = 'INFO'

only_test = False
#checkpoint = f'data/PSData3/model_out/{name}/train_out/model_iter_4500.pth'
# ---> DATASET CONFIG <---
ms_chans = 4
bit_depth = 10
train_set_cfg = dict(
    dataset=dict(
        type='PSDataset',
        #image_dirs=['data/PSData3/Dataset/GF-2/train_low_res'],
        image_dirs=['data/PSData3/Dataset/Dataset/WV-3/train_res'],
        bit_depth=10),
    num_workers=8,
    batch_size=4,
    shuffle=True)
test_set1_cfg = dict(
    dataset=dict(
        type='PSDataset',
        #image_dirs=['data/PSData3/Dataset/GF-2/test_low_res'],
        image_dirs=['data/PSData3/Dataset/Dataset/WV-3/test_res'],
        bit_depth=10),
    num_workers=4,
    batch_size=4,
    shuffle=False)
cuda = True
max_iter = 3000
save_freq = 100
test_freq = 3000
eval_freq = 3000
norm_input = False

# ---> SPECIFIC CONFIG <---
optim_cfg = dict(           #优化器
    core_module=dict(type='Adam', lr=0.0004)
    #Discriminator=dict(type='AdamW', lr=5e-05)
    )
sched_cfg = dict(step_size=200, gamma=0.5)        #策略
loss_cfg = dict(            #LOSS（取一个）
    #QNR_loss=dict(w=1.0),
    #cyc_rec_loss=dict(type='l1', w=0.001),
    #spectral_rec_loss=dict(type='l2', w=0.5),
    #spatial_rec_loss=dict(type='l2', w=0.5)
    all_rec_loss=dict(type='l1',w=1)
    #adv_loss=dict(type='LSGAN', soft_label=True, w=0.001)
    )
'''
model_cfg = dict(           #model
    core_module=dict(       #模型初始化
        hp_filter=True,
        num_blocks=(1, 3, 1),
        n_feats=32,
        norm_type='IN',
        block_type='RCA'),
    Discriminator=dict(n_feats=32, norm_type='IN'),
    to_pan_mode='max')
'''

model_cfg = dict(
    core_module=dict(
        upscale = 4
    )
)
