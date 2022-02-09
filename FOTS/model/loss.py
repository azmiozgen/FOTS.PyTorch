import torch
import torch.nn as nn
from torch.nn import CTCLoss
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)
        intersection = torch.sum(input * target)
        union = torch.sum(input) + torch.sum(target)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)
        intersection = torch.sum(input * target)
        union = torch.sum(input) + torch.sum(target)
        dice_loss = 1.0 - (2. * intersection + self.smooth) / (union + self.smooth)
        bce = F.binary_cross_entropy(input, target, reduction='mean')
        dice_bce_loss = bce + dice_loss
        return dice_bce_loss


class RecognitionLoss(nn.Module):
    def __init__(self):
        super(RecognitionLoss, self).__init__()
        self.ctc_loss = CTCLoss(reduction='mean', zero_infinity=True)

    def forward(self, input, target):
        loss = self.ctc_loss(input[0], target[0], input[1], target[1])
        return loss


class FOTSLoss(nn.Module):

    def __init__(self, config):
        super(FOTSLoss, self).__init__()
        self.mode = config['model']['mode']
        dice_smooth = config['model']['dice_smooth']
        self.detection_loss = DiceBCELoss(smooth=dice_smooth)
        self.recognition_loss = RecognitionLoss()

    def forward(self, y_pred_det, y_true_det, y_pred_recog, y_true_recog):
        if self.mode == 'recognition':
            recognition_loss = self.recognition_loss(y_pred_recog, y_true_recog)
            detection_loss = torch.tensor([0.], device=recognition_loss.device)
        elif self.mode == 'detection':
            detection_loss = self.detection_loss(y_pred_det, y_true_det)
            recognition_loss = torch.tensor([0.], device=detection_loss.device)
        elif self.mode == 'united':
            detection_loss = self.detection_loss(y_pred_det, y_true_det)
            if y_true_recog:
                recognition_loss = self.recognition_loss(y_pred_recog, y_true_recog)
                if recognition_loss < 0:
                    pass  ## TODO

        #recognition_loss = recognition_loss.to(detection_loss.device)
        return detection_loss, recognition_loss

# class DetectionLoss(nn.Module):
#     def __init__(self):
#         super(DetectionLoss, self).__init__()
#         return

#     # def forward(self, y_true_cls, y_pred_cls, training_mask):
#     def forward(self, y_true_cls, y_pred_cls):
#         #classification_loss = self.__dice_coefficient(y_true_cls, y_pred_cls, training_mask)
#         # classification_loss = self.__cross_entropy(y_true_cls, y_pred_cls, training_mask)
#         classification_loss = self.__cross_entropy(y_true_cls, y_pred_cls)
#         # scale classification loss to match the iou loss part
#         classification_loss *= 0.01

#         return torch.mean(y_true_cls), classification_loss

#     # def __dice_coefficient(self, y_true_cls, y_pred_cls,
#     #                      training_mask):
#     def __dice_coefficient(self, y_true_cls, y_pred_cls, smooth=1.):
#         '''
#         dice loss
#         :param y_true_cls:
#         :param y_pred_cls:
#         :param training_mask:
#         :return:
#         '''
#         # intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
#         # union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
#         y_pred_cls = y_pred_cls.view(-1)
#         y_true_cls = y_true_cls.view(-1)
#         intersection = torch.sum(y_true_cls * y_pred_cls)
#         union = torch.sum(y_true_cls) + torch.sum(y_pred_cls)
#         dice = (2 * intersection + smooth) / (union + smooth)

#         return 1.0 - dice

#     # def __cross_entropy(self, y_true_cls, y_pred_cls, training_mask):
#     def __cross_entropy(self, y_true_cls, y_pred_cls):
#         #import ipdb; ipdb.set_trace()
#         # return torch.nn.functional.binary_cross_entropy(y_pred_cls * training_mask, (y_true_cls*training_mask))
#         return torch.nn.functional.binary_cross_entropy(y_pred_cls, y_true_cls)