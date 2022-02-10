import numpy as np
import pretrainedmodels as pm
import torch
import torch.nn as nn
import torch.optim as optim

from ..base import BaseModel
from .modules import shared_conv
from .modules.roi_rotate import ROIRotate
from .modules.crnn import CRNN
from .keys import keys
from ..utils.bbox import Toolbox
from ..data_loader.datautils import is_bbox8_ok


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
        image_files, image, boxes, mapping = input

        if image.is_cuda:
            device = image.get_device()
        else:
            device = torch.device('cpu')

        feature_map = self.sharedConv.forward(image)
        score_map = self.detector(feature_map)

        if self.training:
            boxes_norm = boxes[:, :8] / 4
            rois, lengths, indices = self.get_cropped_padded_features(feature_map, boxes_norm, image_files)
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
                if not is_bbox8_ok(bb):
                    h, w = s.shape
                    bb = np.array([0, 0, w, 0, w, h, 0, h, 1])
                pred_mapping.append(i)
                pred_boxes.append(bb)

            if len(pred_mapping) > 0:
                pred_boxes = np.array(pred_boxes).astype(np.float32)
                pred_mapping = np.array(pred_mapping)
                pred_boxes_norm = pred_boxes[:, :8] / 4
                rois, lengths, indices = self.get_cropped_padded_features(feature_map, pred_boxes_norm, image_files)
            else:
                return score_map, (None, None), pred_boxes, pred_mapping, None

        lengths = torch.tensor(lengths).to(device)
        preds = self.recognizer(rois, lengths)
        preds = preds.permute(1, 0, 2) # B, T, C -> T, B, C

        return score_map, (preds, lengths), pred_boxes, pred_mapping, indices

    def get_cropped_padded_features(self, feature_map, boxes, image_files):
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
            assert is_bbox8_ok(boxes[i]), 'BBOX8 NOT OK ' + str(boxes[i]) + ' ' + str(image_files[i])
            w = boxes[i, 2] - boxes[i, 0]
            h = boxes[i, 5] - boxes[i, 1]
            cropped_feature = feature_map[i, :, boxes[i, 1]:boxes[i, 5], boxes[i, 0]:boxes[i, 2]] # [B, :, y1:y3, x1:x2]
            cropped_feature = nn.ZeroPad2d((0, max(0, max_width - w), 0, max(0, max_height - h)))(cropped_feature)
            cropped_padded_features.append(cropped_feature)
        cropped_padded_features = torch.stack(cropped_padded_features, dim=0)

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
        n_hidden = self.config['model']['recognizer_n_hidden']
        self.crnn = CRNN(32, nclass, n_hidden, dropout=dropout)

    def forward(self, rois, lengths):
        return self.crnn(rois, lengths)


class Detector(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        # dropout = self.config['model']['dropout']
        self.scoreMap = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            # nn.Dropout2d(dropout, inplace=True),
        )
        self.input_size = config['data_loader']['input_size']

    def forward(self, *input):
        final,  = input

        score = self.scoreMap(final)
        score = torch.sigmoid(score)

        return score
