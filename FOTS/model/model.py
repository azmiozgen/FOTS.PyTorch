from ..base import BaseModel
import torch.nn as nn
import torch
import math
from .modules import shared_conv
from .modules.roi_rotate import ROIRotate
from .modules.crnn import CRNN
from .keys import keys
import pretrainedmodels as pm
import torch.optim as optim
from ..utils.bbox import Toolbox
import numpy as np


class FOTSModel:

    def __init__(self, config):

        self.mode = config['model']['mode']
        self.score_map_threshold = config['model']['score_map_threshold']

        # bbNet =  pm.__dict__['resnet50'](pretrained='imagenet') # resnet50 in paper
        # bbNet =  pm.__dict__['resnet34'](pretrained='imagenet')
        bbNet =  pm.__dict__['resnet18'](pretrained='imagenet')
        self.sharedConv = shared_conv.SharedConv(bbNet, config)

        nclass = len(keys) + 1
        self.recognizer = Recognizer(nclass, config)
        self.detector = Detector(config)
        self.roirotate = ROIRotate()

    def parallelize(self):
        self.sharedConv = torch.nn.DataParallel(self.sharedConv)
        self.recognizer = torch.nn.DataParallel(self.recognizer)
        self.detector = torch.nn.DataParallel(self.detector)
        #self.roirotate = torch.nn.DataParallel(self.roirotate)

    def to(self, device):
        self.sharedConv = self.sharedConv.to(device)
        self.detector = self.detector.to(device)
        self.recognizer = self.recognizer.to(device)

    def summary(self):
        self.sharedConv.summary()
        self.detector.summary()
        self.recognizer.summary()

    def optimize(self, optimizer_type, params):
        optimizer = getattr(optim, optimizer_type)(
            [
                {'params': self.sharedConv.parameters()},
                {'params': self.detector.parameters()},
                {'params': self.recognizer.parameters()},
            ],
            **params
        )
        return optimizer

    def train(self):
        self.sharedConv.train()
        self.detector.train()
        self.recognizer.train()

    def eval(self):
        self.sharedConv.eval()
        self.detector.eval()
        self.recognizer.eval()

    def state_dict(self):
        sd = {
            '0': self.sharedConv.state_dict(),
            '1': self.detector.state_dict(),
            '2': self.recognizer.state_dict()
        }
        return sd

    def load_state_dict(self, sd):
        self.sharedConv.load_state_dict(sd['0'])
        self.detector.load_state_dict(sd['1'])
        self.recognizer.load_state_dict(sd['2'])

    @property
    def training(self):
        return self.sharedConv.training and self.detector.training and self.recognizer.training

    def forward(self, *input):
        '''

        :param input:
        :return:
        '''
        image, boxes, mapping = input

        if image.is_cuda:
            device = image.get_device()
        else:
            device = torch.device('cpu')

        feature_map = self.sharedConv.forward(image)
        score_map = self.detector(feature_map)

        if self.training:
            # rois, lengths, indices = self.roirotate(feature_map, boxes[:, :8], mapping)
            boxes_norm = boxes[:, :8] / 4
            rois, lengths, indices = self.get_cropped_padded_features(feature_map, boxes_norm)
            pred_mapping = mapping
            pred_boxes = boxes
        else:
            score = score_map.permute(0, 2, 3, 1)
            score = score.detach().cpu().numpy()

            pred_boxes = []
            pred_mapping = []
            for i in range(score.shape[0]):
                s = score[i, :, :, 0]
                bb = Toolbox.detect(score_map=s, score_map_thresh=self.score_map_threshold)

                # pred_mapping.append(np.array([i] * bb_size))
                pred_mapping.append(i)
                pred_boxes.append(bb)

            if len(pred_mapping) > 0:
                pred_boxes = np.array(pred_boxes).astype(np.float32)
                pred_mapping = np.array(pred_mapping)
                # print('FEATURE MAP', feature_map.shape)
                # print('PRED BOXES', pred_boxes.shape)
                # print('PRED MAPPING', pred_mapping.shape)
                # rois, lengths, indices = self.roirotate(feature_map, pred_boxes[:, :8], pred_mapping)
                pred_boxes_norm = pred_boxes[:, :8] / 4
                rois, lengths, indices = self.get_cropped_padded_features(feature_map, pred_boxes_norm)
                # print('ROIS', rois.shape)
                # print('LENGTHS', lengths.shape, lengths)
                # print('INDICES', indices.shape, indices)
            else:
                # return score_map, geo_map, (None, None), pred_boxes, pred_mapping, None
                return score_map, (None, None), pred_boxes, pred_mapping, None

        lengths = torch.tensor(lengths).to(device)
        preds = self.recognizer(rois, lengths)
        preds = preds.permute(1, 0, 2) # B, T, C -> T, B, C

        # return score_map, geo_map, (preds, lengths), pred_boxes, pred_mapping, indices
        return score_map, (preds, lengths), pred_boxes, pred_mapping, indices

    def get_cropped_padded_features(self, feature_map, boxes, ):
        '''
        feature_map: B, C, H, W
        boxes: B, 8 (x1, y1, x2, y2, x3, y3, x4, y4)
        pred_mapping: B
        '''
        n_batches = feature_map.shape[0]
        boxes = boxes.astype(np.int)

        ## Get max height and width in boxes
        max_height = feature_map.shape[2]
        max_width = max(boxes[:, 2] - boxes[:, 0])
        cropped_padded_features = []

        for i in range(n_batches):
            w = boxes[i, 2] - boxes[i, 0]
            h = boxes[i, 5] - boxes[i, 1]
            cropped_feature = feature_map[i, :, boxes[i, 1]:boxes[i, 5], boxes[i, 0]:boxes[i, 2]] # [B, :, y1:y3, x1:x2]
            cropped_feature = nn.ZeroPad2d((0, max(0, max_width - w), 0, max(0, max_height - h)))(cropped_feature)
            cropped_padded_features.append(cropped_feature)
        cropped_padded_features = torch.stack(cropped_padded_features, dim=0)
        # print('CROPPED PADDED FEATURES', cropped_padded_features.shape)

        lengths = boxes[:, 2] - boxes[:, 0]
        indices = np.argsort(lengths) # sort images by its width cause pack padded tensor needs it
        indices = indices[::-1].copy() # descending order
        lengths = lengths[indices]
        cropped_padded_features = cropped_padded_features[indices]

        return cropped_padded_features, lengths, indices

class Recognizer(BaseModel):

    def __init__(self, nclass, config):
        super().__init__(config)
        dropout = self.config['model']['dropout']
        self.crnn = CRNN(32, nclass, 256, dropout=dropout)

    def forward(self, rois, lengths):
        return self.crnn(rois, lengths)


class Detector(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        dropout = self.config['model']['dropout']
        # self.scoreMap = nn.Conv2d(32, 1, kernel_size = 1)
        self.scoreMap = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Dropout2d(dropout, inplace=True),
        )
        # self.geoMap = nn.Conv2d(32, 4, kernel_size = 1)
        # self.angleMap = nn.Conv2d(32, 1, kernel_size = 1)
        self.input_size = config['data_loader']['input_size']

    def forward(self, *input):
        final,  = input

        score = self.scoreMap(final)
        score = torch.sigmoid(score)

        # geoMap = self.geoMap(final)
        # geoMap = torch.sigmoid(geoMap) * self.input_size

        # angleMap = self.angleMap(final)
        # angleMap = (torch.sigmoid(angleMap) - 0.5) * math.pi / 2

        # geometry = torch.cat([geoMap, angleMap], dim=1)

        # return score, geometry
        return score
