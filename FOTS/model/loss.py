import torch
import torch.nn as nn
from torch.nn import CTCLoss


class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
        return

    # def forward(self, y_true_cls, y_pred_cls,
    #             y_true_geo, y_pred_geo,
    #             training_mask):
    def forward(self, y_true_cls, y_pred_cls, training_mask):
        #classification_loss = self.__dice_coefficient(y_true_cls, y_pred_cls, training_mask)

        classification_loss = self.__cross_entropy(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01

        return torch.mean(y_true_cls * training_mask), classification_loss

    def __dice_coefficient(self, y_true_cls, y_pred_cls,
                         training_mask):
        '''
        dice loss
        :param y_true_cls:
        :param y_pred_cls:
        :param training_mask:
        :return:
        '''
        eps = 1e-5
        intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
        union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
        loss = 1. - (2 * intersection / union)

        return loss

    def __cross_entropy(self, y_true_cls, y_pred_cls, training_mask):
        #import ipdb; ipdb.set_trace()
        return torch.nn.functional.binary_cross_entropy(y_pred_cls*training_mask, (y_true_cls*training_mask))


class RecognitionLoss(nn.Module):

    def __init__(self):
        super(RecognitionLoss, self).__init__()
        self.ctc_loss = CTCLoss(reduction='mean', zero_infinity=True) # pred, pred_len, labels, labels_len

    def forward(self, *input):
        gt, pred = input[0], input[1]
        loss = self.ctc_loss(pred[0], gt[0], pred[1], gt[1])
        return loss


class FOTSLoss(nn.Module):

    def __init__(self, config):
        super(FOTSLoss, self).__init__()
        self.mode = config['model']['mode']
        self.detectionLoss = DetectionLoss()
        self.recogitionLoss = RecognitionLoss()

    # def forward(self, y_true_cls, y_pred_cls,
    #             y_true_geo, y_pred_geo,
    #             y_true_recog, y_pred_recog,
    #             training_mask):
    def forward(self, y_true_cls, y_pred_cls,
                y_true_recog, y_pred_recog,
                training_mask):
        if self.mode == 'recognition':
            recognition_loss = self.recogitionLoss(y_true_recog, y_pred_recog)
            reg_loss = torch.tensor([0.], device=recognition_loss.device)
            cls_loss = torch.tensor([0.], device=recognition_loss.device)
        elif self.mode == 'detection':
            # reg_loss, cls_loss = self.detectionLoss(y_true_cls, y_pred_cls,
            #                                     y_true_geo, y_pred_geo, training_mask)
            reg_loss, cls_loss = self.detectionLoss(y_true_cls, y_pred_cls, training_mask)
            recognition_loss = torch.tensor([0.], device=reg_loss.device)
        elif self.mode == 'united':
            # reg_loss, cls_loss = self.detectionLoss(y_true_cls, y_pred_cls,
            #                                     y_true_geo, y_pred_geo, training_mask)
            reg_loss, cls_loss = self.detectionLoss(y_true_cls, y_pred_cls, training_mask)
            if y_true_recog:
                recognition_loss = self.recogitionLoss(y_true_recog, y_pred_recog)
                if recognition_loss < 0:
                    # import ipdb; ipdb.set_trace()
                    pass

        #recognition_loss = recognition_loss.to(detection_loss.device)
        return reg_loss, cls_loss, recognition_loss
