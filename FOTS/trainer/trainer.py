import time

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
    def __init__(self, model, loss, metrics, resume, config, config_file,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, resume, config, config_file, train_logger)
        self.batch_size = data_loader.batch_size
        self.config = config
        self.config_file = config_file
        self.data_loader = data_loader
        self.label_converter = strLabelConverter(keys)
        self.len_data_loader = len(data_loader)
        self.len_valid_data_loader = len(valid_data_loader)
        self.log_step = int(np.sqrt(self.batch_size))
        self.mode = self.config['model']['mode']
        self.valid = True if valid_data_loader is not None else False
        self.valid_data_loader = valid_data_loader

    def _to_device(self, *tensors):
        t = []
        for __tensors in tensors:
            t.append(__tensors.to(self.device))
        return t

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        if self.mode == 'united':
            self.model.train()
        elif self.mode == 'detection':
            self.model.train_detector()
        elif self.mode == 'recognition':
            self.model.train_recognizer()
        else:
            self.model.train()

        total_loss, total_det_loss, total_rec_loss = 0, 0, 0
        text_accuracy = 0.0
        value_error = 0.0
        start_time = time.time()
        for batch_idx, gt in enumerate(self.data_loader):
            try:
                image_files, _image, score_map, transcriptions, boxes, mapping = gt
                image, score_map = self._to_device(_image.clone(), score_map.clone())

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

                labels, label_lengths = self.label_converter.encode(transcriptions.tolist())
                labels = labels.to(self.device)
                label_lengths = label_lengths.to(self.device)
                recog = (labels, label_lengths)

                detection_loss, recognition_loss = self.loss(pred_score_map, score_map, pred_recog, recog)
                loss = detection_loss + recognition_loss
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_det_loss += detection_loss.item()
                total_rec_loss += recognition_loss.item()
                pred_transcriptions = []
                if len(pred_mapping) > 0:
                    pred, lengths = pred_recog
                    _, pred = pred.max(2)
                    for i in range(lengths.numel()):
                        l = lengths[i]
                        p = pred[:l, i]
                        t = self.label_converter.decode(p, l)
                        pred_transcriptions.append(t)
                    pred_transcriptions = np.array(pred_transcriptions)

                if batch_idx == 0:
                    print('Training gt transcriptions:', transcriptions[:8])
                    print('Training pred transcriptions:', pred_transcriptions[:8])

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

        total_time = round(time.time() - start_time, 2)
        avg_loss = total_loss / self.len_data_loader
        avg_det_loss = total_det_loss / self.len_data_loader
        avg_rec_loss = total_rec_loss / self.len_data_loader
        if self.verbosity >= 2:
            self.logger.info(\
                'Train: Epoch: {} [{} samples, {} seconds] Loss: {:.6f} Detection Loss: {:.6f} Recognition Loss: {:.6f}'.format(
                    epoch,
                    len(self.data_loader.dataset),
                    total_time,
                    avg_loss, avg_det_loss, avg_rec_loss))

        log = {
            'loss': avg_loss,
            'det_loss': avg_det_loss,
            'rec_loss': avg_rec_loss,
            'text_accuracy': text_accuracy / self.len_data_loader,
            'value_error': value_error / self.len_data_loader,
            'time': total_time
        }

        if self.valid:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        """
        self.model.eval()
        total_loss, total_det_loss, total_rec_loss = 0, 0, 0
        text_accuracy = 0.0
        value_error = 0.0
        start_time = time.time()
        with torch.no_grad():
            for batch_idx, gt in enumerate(self.valid_data_loader):
                try:
                    image_files, _image, score_map, transcriptions, boxes, mapping = gt
                    image, score_map = self._to_device(_image.clone(), score_map.clone())

                    pred_score_map, pred_recog, pred_boxes, pred_mapping, indices = \
                            self.model.forward(image_files, image, boxes, mapping)
                    transcriptions = transcriptions[indices]
                    image_files = np.array(image_files)[indices]
                    image_visual = _image[indices]
                    pred_boxes = pred_boxes[indices]
                    pred_mapping = pred_mapping[indices]
                    pred_score_map = pred_score_map[indices]
                    score_map = score_map[indices]

                    labels, label_lengths = self.label_converter.encode(transcriptions.tolist())
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
                            t = self.label_converter.decode(p, l)
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

                    detection_loss, recognition_loss = self.loss(pred_score_map, score_map, pred_recog, recog)
                    loss = detection_loss + recognition_loss
                    total_loss += loss.item()
                    total_det_loss += detection_loss.item()
                    total_rec_loss += recognition_loss.item()

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

        total_time = round(time.time() - start_time, 2)
        avg_loss = total_loss / self.len_valid_data_loader
        avg_det_loss = total_det_loss / self.len_valid_data_loader
        avg_rec_loss = total_rec_loss / self.len_valid_data_loader
        if self.verbosity >= 2:
            self.logger.info(\
                'Val: Epoch: {} [{} samples, {:.2f} seconds] Loss: {:.6f} Detection Loss: {:.6f} Recognition Loss: {:.6f}'.format(
                    epoch,
                    len(self.valid_data_loader.dataset),
                    total_time,
                    avg_loss, avg_det_loss, avg_rec_loss))

        return {
            'val_loss': avg_loss,
            'val_det_loss': avg_det_loss,
            'val_rec_loss': avg_rec_loss,
            'val_text_accuracy': text_accuracy / self.len_valid_data_loader,
            'val_value_error': value_error / self.len_valid_data_loader,
            'val_time': total_time
        }
