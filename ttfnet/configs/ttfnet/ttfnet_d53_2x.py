# model settings
model = dict(
    type='TTFNet',
    pretrained='./pretrain/darknet53.pth',
    backbone=dict( #transforms an image to feature maps
        type='DarknetV3',
        layers=[1, 2, 8, 8, 4],
        inplanes=[3, 32, 64, 128, 256, 512],
        planes=[32, 64, 128, 256, 512, 1024],
        norm_cfg=dict(type='BN'),
        out_indices=(1, 2, 3, 4),
        frozen_stages=1,
        norm_eval=False),
    neck=None, #connects the backbone and heads
    bbox_head=dict( #operates on dense locations of feature maps
        type='TTFHead',
        inplanes=(128, 256, 512, 1024),
        head_conv=128,
        wh_conv=64,
        hm_head_conv_num=2,
        wh_head_conv_num=2,
        num_classes=80,
        wh_offset_base=16,
        wh_agnostic=True,
        wh_gaussian=True,
        shortcut_cfg=(1, 2, 3),
        norm_cfg=dict(type='BN'),
        alpha=0.54,
        hm_weight=1.,
        wh_weight=5.))
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(
    vis_every_n_iters=100,
    debug=False)
test_cfg = dict(
    score_thr=0.01,
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/rladkdud31/data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=12,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'new_json.json', #'annotations/instances_train2014.json',
        img_prefix=data_root + 'input_image/', #+ 'images/train2014/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        resize_keep_ratio=False),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'new_json.json', #+ 'annotations/instances_val2014.json',
        img_prefix=data_root + 'nput_image/', # + 'images/val2014/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        resize_keep_ratio=False),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'new_json.json', # + 'annotations/instances_val2014.json',
        img_prefix=data_root + 'nput_image/', # + 'images/val2014/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False))
# optimizer
optimizer = dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0004,
                 paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    step=[18, 22])
checkpoint_config = dict(interval=4)
bbox_head_hist_config = dict(
    model_type=['ConvModule', 'DeformConvPack'],
    sub_modules=['bbox_head'],
    save_every_n_steps=500)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
# runtime settings
total_epochs = 24
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/ttfnet53_2x'
load_from = 'pretrain/ttfnet53_2x-b381dd.pth'
resume_from = None
workflow = [('train', 1)]
