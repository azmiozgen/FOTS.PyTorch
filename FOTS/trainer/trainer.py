import os

import numpy as np
import torch
import torchvision
from ..base import BaseTrainer
from ..utils.bbox import Toolbox
from ..model.keys import keys
from ..utils.util import strLabelConverter
from ..data_loader.datautils import draw_text_tensor

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, toolbox: Toolbox, valid_data_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.len_data_loader = len(self.data_loader)
        self.len_valid_data_loader = len(self.valid_data_loader)
        self.valid = True if self.valid_data_loader is not None else False
        self.log_step = int(np.sqrt(self.batch_size))
        self.toolbox = toolbox
        self.labelConverter = strLabelConverter(keys)

    def _to_device(self, *tensors):
        t = []
        for __tensors in tensors:
            t.append(__tensors.to(self.device))
        return t

    def _eval_metrics(self, pred, gt):
        precision, recall, hmean = self.metrics[0](pred, gt)
        return np.array([precision, recall, hmean])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss, total_iou_loss, total_cls_loss, total_rec_loss = 0, 0, 0, 0
        total_metrics = np.zeros(3) # precision, recall, hmean
        text_accuracy = 0.0
        value_error = 0.0
        for batch_idx, gt in enumerate(self.data_loader):
            try:
                image_files, _image, score_map, training_mask, transcriptions, boxes, mapping = gt
                image, score_map, training_mask = self._to_device(_image.clone(), score_map.clone(), training_mask.clone())

                self.optimizer.zero_grad()
                pred_score_map, pred_recog, pred_boxes, pred_mapping, indices = \
                        self.model.forward(image_files, image, boxes, mapping)
                transcriptions = transcriptions[indices]
                image_files = np.array(image_files)[indices]
                image_visual = _image[indices]
                pred_boxes = pred_boxes[indices]
                pred_mapping = pred_mapping[indices]
                pred_score_map = pred_score_map[indices]
                score_map = score_map[indices]

                labels, label_lengths = self.labelConverter.encode(transcriptions.tolist())
                labels = labels.to(self.device)
                label_lengths = label_lengths.to(self.device)
                recog = (labels, label_lengths)

                iou_loss, cls_loss, rec_loss = self.loss(score_map, pred_score_map, recog, pred_recog, training_mask)
                loss = iou_loss + cls_loss + rec_loss
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_iou_loss += iou_loss.item()
                total_cls_loss += cls_loss.item()
                total_rec_loss += rec_loss.item()
                pred_transcriptions = []
                if len(pred_mapping) > 0:
                    pred, lengths = pred_recog
                    _, pred = pred.max(2)
                    for i in range(lengths.numel()):
                        l = lengths[i]
                        p = pred[:l, i]
                        t = self.labelConverter.decode(p, l)
                        pred_transcriptions.append(t)
                    pred_transcriptions = np.array(pred_transcriptions)

                if batch_idx == 0:
                    print('Training gt transcriptions:', transcriptions)
                    print('Training pred transcriptions:', pred_transcriptions)
                    print('Training files', list(map(os.path.basename, image_files)))

                ## Write summary writer images and text
                if epoch % self.save_freq == 0 and batch_idx == 0:
                    step = epoch * self.len_data_loader + batch_idx

                    image_grid = torchvision.utils.make_grid(image_visual, normalize=True, scale_each=True)
                    score_map_grid = torchvision.utils.make_grid(score_map)
                    pred_score_map_grid = torchvision.utils.make_grid(pred_score_map.double())

                    gt_transcriptions_tensor = draw_text_tensor(transcriptions)
                    gt_transcriptions_grid = torchvision.utils.make_grid(gt_transcriptions_tensor,
                            normalize=True, value_range=(0, 1))

                    pred_transcriptions_tensor = draw_text_tensor(pred_transcriptions)
                    pred_transcriptions_grid = torchvision.utils.make_grid(pred_transcriptions_tensor,
                            normalize=True, value_range=(0, 1))

                    self.summary_writer.add_image('train_images', image_grid, step)
                    self.summary_writer.add_image('train_gt_masks', score_map_grid, step)
                    self.summary_writer.add_image('train_pred_masks', pred_score_map_grid, step)
                    self.summary_writer.add_image('train_gt_transcriptions', gt_transcriptions_grid, step)
                    self.summary_writer.add_image('train_pred_transcriptions', pred_transcriptions_grid, step)

                total_metrics += self._eval_metrics((pred_boxes, pred_transcriptions, image_files),
                                                        (boxes, transcriptions, image_files))

                ## Transcripton accuracy
                text_accuracy += np.mean(transcriptions == pred_transcriptions)

                ## Value error
                try:
                    batch_value_error = 0.0
                    for tr, pred_tr in zip(transcriptions, pred_transcriptions):
                        tr1, tr2 = tr.split(',')
                        tr_value = int(tr1) + int(tr2) / 100.0
                        pred_tr1, pred_tr2 = pred_tr.split(',')
                        pred_tr_value = int(pred_tr1) + int(pred_tr2) / 100.0
                        batch_value_error += abs(tr_value - pred_tr_value) / (tr_value + 1e-10)
                    value_error += (batch_value_error / len(transcriptions))
                except ValueError:
                    value_error += 1.0

            except Exception as e:
                print(e, 'Training failed')
                raise

        avg_loss = total_loss / self.len_data_loader
        if self.verbosity >= 2:
            avg_iou_loss = total_iou_loss / self.len_data_loader
            avg_cls_loss = total_cls_loss / self.len_data_loader
            avg_rec_loss = total_rec_loss / self.len_data_loader
            self.logger.info(\
                'Train: Epoch: {} [{} samples] Loss: {:.6f} IOU Loss: {:.6f} CLS Loss: {:.6f} Recognition Loss: {:.6f}'.format(
                    epoch,
                    len(self.data_loader.dataset),
                    avg_loss, avg_iou_loss, avg_cls_loss, avg_rec_loss))

        log = {
            'loss': avg_loss,
            'precision': total_metrics[0] / self.len_data_loader,
            'recall': total_metrics[1] / self.len_data_loader,
            'hmean': total_metrics[2] / self.len_data_loader,
            'text_accuracy': text_accuracy / self.len_data_loader,
            'value_error': value_error / self.len_data_loader
        }

        if self.valid:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_metrics = np.zeros(3)
        total_loss, total_iou_loss, total_cls_loss, total_rec_loss = 0, 0, 0, 0
        total_val_metrics = np.zeros(3) # precision, recall, hmean
        text_accuracy = 0.0
        value_error = 0.0
        with torch.no_grad():
            for batch_idx, gt in enumerate(self.valid_data_loader):
                try:
                    image_files, _image, score_map, training_mask, transcriptions, boxes, mapping = gt
                    image, score_map, training_mask = self._to_device(_image.clone(), score_map.clone(), training_mask.clone())

                    pred_score_map, pred_recog, pred_boxes, pred_mapping, indices = \
                            self.model.forward(image_files, image, boxes, mapping)
                    transcriptions = transcriptions[indices]
                    image_files = np.array(image_files)[indices]
                    image_visual = _image[indices]
                    pred_boxes = pred_boxes[indices]
                    pred_mapping = pred_mapping[indices]
                    pred_score_map = pred_score_map[indices]
                    score_map = score_map[indices]

                    labels, label_lengths = self.labelConverter.encode(transcriptions.tolist())
                    labels = labels.to(self.device)
                    label_lengths = label_lengths.to(self.device)
                    recog = (labels, label_lengths)

                    pred_transcriptions = []
                    if len(pred_mapping) > 0:
                        pred, lengths = pred_recog
                        _, pred = pred.max(2)
                        for i in range(lengths.numel()):
                            l = lengths[i]
                            p = pred[:l, i]
                            t = self.labelConverter.decode(p, l)
                            pred_transcriptions.append(t)
                        pred_transcriptions = np.array(pred_transcriptions)

                    if batch_idx == 0:
                        print('Validation gt transcriptions:', transcriptions)
                        print('Validation pred transcriptions:', pred_transcriptions)

                    ## Write summary writer images and text
                    if epoch % self.save_freq == 0 and batch_idx == 0:
                        step = epoch * self.len_valid_data_loader + batch_idx

                        image_grid = torchvision.utils.make_grid(image_visual, normalize=True, scale_each=True)
                        score_map_grid = torchvision.utils.make_grid(score_map)
                        pred_score_map_grid = torchvision.utils.make_grid(pred_score_map.double())

                        gt_transcriptions_tensor = draw_text_tensor(transcriptions)
                        gt_transcriptions_grid = torchvision.utils.make_grid(gt_transcriptions_tensor,
                                normalize=True, value_range=(0, 1))

                        pred_transcriptions_tensor = draw_text_tensor(pred_transcriptions)
                        pred_transcriptions_grid = torchvision.utils.make_grid(pred_transcriptions_tensor,
                                normalize=True, value_range=(0, 1))

                        self.summary_writer.add_image('val_images', image_grid, step)
                        self.summary_writer.add_image('val_gt_masks', score_map_grid, step)
                        self.summary_writer.add_image('val_pred_masks', pred_score_map_grid, step)
                        self.summary_writer.add_image('val_gt_transcriptions', gt_transcriptions_grid, step)
                        self.summary_writer.add_image('val_pred_transcriptions', pred_transcriptions_grid, step)

                    iou_loss, cls_loss, rec_loss = self.loss(score_map, pred_score_map, recog, pred_recog, training_mask)
                    loss = iou_loss + cls_loss + rec_loss
                    total_loss += loss.item()
                    total_iou_loss += iou_loss.item()
                    total_cls_loss += cls_loss.item()
                    total_rec_loss += rec_loss.item()

                    total_val_metrics += self._eval_metrics((pred_boxes, pred_transcriptions, image_files),
                                                            (boxes, transcriptions, image_files))

                    ## Transcripton accuracy
                    text_accuracy += np.mean(transcriptions == pred_transcriptions)

                    ## Value error
                    try:
                        batch_value_error = 0.0
                        for tr, pred_tr in zip(transcriptions, pred_transcriptions):
                            tr1, tr2 = tr.split(',')
                            tr_value = int(tr1) + int(tr2) / 100.0
                            pred_tr1, pred_tr2 = pred_tr.split(',')
                            pred_tr_value = int(pred_tr1) + int(pred_tr2) / 100.0
                            batch_value_error += abs(tr_value - pred_tr_value) / (tr_value + 1e-10)
                        value_error += (batch_value_error / len(transcriptions))
                    except:
                        value_error += 1.0

                except Exception as e:
                    print(e, 'Validation failed')
                    # raise

        avg_loss = total_loss / self.len_valid_data_loader
        if self.verbosity >= 2:
            avg_iou_loss = total_iou_loss / self.len_valid_data_loader
            avg_cls_loss = total_cls_loss / self.len_valid_data_loader
            avg_rec_loss = total_rec_loss / self.len_valid_data_loader
            self.logger.info(\
                'Val: Epoch: {} [{} samples] Loss: {:.6f} IOU Loss: {:.6f} CLS Loss: {:.6f} Recognition Loss: {:.6f}'.format(
                    epoch,
                    len(self.valid_data_loader.dataset),
                    avg_loss, avg_iou_loss, avg_cls_loss, avg_rec_loss))

        return {
            'val_loss': avg_loss,
            'val_precision': total_val_metrics[0] / self.len_valid_data_loader,
            'val_recall': total_val_metrics[1] / self.len_valid_data_loader,
            'val_hmean': total_val_metrics[2] / self.len_valid_data_loader,
            'val_text_accuracy': text_accuracy / self.len_valid_data_loader,
            'val_value_error': value_error / self.len_valid_data_loader
        }
