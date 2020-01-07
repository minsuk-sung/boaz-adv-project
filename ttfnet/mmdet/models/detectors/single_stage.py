import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector


@DETECTORS.register_module
class SingleStageDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        # print('len(x):',len(x)) #4
        # print('x:',x[3].shape)
        # x[0].shape : (12, 128, 128, 128)
        # x[1].shape : (12, 256, 64, 64)
        # x[2].shape : (12, 512, 32, 32)
        # x[3].shape : (12, 1024, 16, 16)
        # (128,256,512,1024) : ttfnet_d53_2x.py, bbox_head - inplanes
        # 12 : batch_size(imgs_per_gpu)

        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # print('len(outs):',len(outs[0])) # 12(batch_size)
        # print('outs:',outs[1][0].shape) #(4,128,128)
        # print('outs:',outs[0][0].shape) # (80,128,128)
        
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        # print('len_loss2:',len(loss_inputs[0])) # 12
        # print('loss_inputs1:',loss_inputs[0][0].shape) # (80,128,128)
        # print('loss_inputs3:',loss_inputs[1][0].shape) # (4,128,128)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
