import time
from collections import OrderedDict

import numpy as np
import torch
import torchvision

from ..base import BaseTrainer
from ..data_loader.datautils import draw_text_tensor
from ..model.keys import keys
from ..model.metric import get_mean_char_similarity, mae
from ..utils.util import strLabelConverter

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

    def _resume_checkpoint(self, resume_path):
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except RuntimeError:  ## Load nn.DataParallel
            for s in ['0', '1', '2']:  ## SharedConv, Detector, Recognizer
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'][s].items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                checkpoint['state_dict'][s] = new_state_dict
            self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.with_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(torch.device('cuda'))
        self.train_logger = checkpoint['logger']
        #self.config = checkpoint['config']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

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
        text_accuracy, text_accuracy_wo_decimal, char_similarity, value_mae = 0.0, 0.0, 0.0, 0.0
        start_time = time.time()
        for batch_idx, gt in enumerate(self.data_loader):
            try:
                image_files, _image, score_map, transcriptions, boxes = gt
                image, score_map = self._to_device(_image.clone(), score_map.clone())

                self.optimizer.zero_grad()
                pred_score_map, pred_recog, pred_boxes, indices = self.model.forward(image_files, image, boxes)
                gt_transcriptions = transcriptions[indices]
                image_files = np.array(image_files)[indices]
                image_visual = _image[indices]
                pred_boxes = pred_boxes[indices]
                pred_score_map = pred_score_map[indices]
                score_map = score_map[indices]

                labels, label_lengths = self.label_converter.encode(gt_transcriptions.tolist())
                labels = labels.to(self.device)
                label_lengths = label_lengths.to(self.device)
                recog = (labels, label_lengths)

                ## Get loss and optimize
                detection_loss, recognition_loss = self.loss(pred_score_map, score_map, pred_recog, recog)
                loss = detection_loss + recognition_loss
                loss.backward()
                self.optimizer.step()

                ## Update total losses
                total_loss += loss.item()
                total_det_loss += detection_loss.item()
                total_rec_loss += recognition_loss.item()

                ## Get transcription predictions
                pred_transcriptions = []
                pred, lengths = pred_recog
                _, pred = pred.max(2)
                for i in range(lengths.numel()):
                    l = lengths[i]
                    p = pred[:l, i]
                    t = self.label_converter.decode(p, l)
                    pred_transcriptions.append(t)
                pred_transcriptions = np.array(pred_transcriptions)

                if batch_idx == 0:
                    print('Training gt transcriptions:', gt_transcriptions[:8])
                    print('Training pred transcriptions:', pred_transcriptions[:8])

                ## Write summary writer images and text
                if epoch % self.save_freq == 0 and batch_idx == 0:
                    step = epoch * self.len_data_loader + batch_idx

                    image_grid = torchvision.utils.make_grid(image_visual, normalize=True, scale_each=True)
                    score_map_grid = torchvision.utils.make_grid(score_map)
                    pred_score_map_grid = torchvision.utils.make_grid(pred_score_map.double())

                    gt_transcriptions_tensor = draw_text_tensor(gt_transcriptions)
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
                text_accuracy += np.mean(gt_transcriptions == pred_transcriptions)

                ## Text without decimal part accuracy
                gt_trans_wo_decimal = np.array(list(map(lambda s: s.split(',')[0], gt_transcriptions))).astype(str)
                pred_trans_wo_decimal = np.array(list(map(lambda s: s.split(',')[0], pred_transcriptions))).astype(str)
                text_accuracy_wo_decimal += np.mean(gt_trans_wo_decimal == pred_trans_wo_decimal)

                ## Char accuracy by SequenceMatcher
                char_similarity += get_mean_char_similarity(pred_transcriptions, gt_transcriptions)

                ## Value MSE
                gt_transcription_values = np.array(list(map(lambda s: self.get_transcription_value(s), gt_transcriptions))).astype(np.float32)
                pred_transcription_values = np.array(list(map(lambda s: self.get_transcription_value(s), pred_transcriptions))).astype(np.float32)
                value_mae += mae(gt_transcription_values, pred_transcription_values)

            except Exception as e:
                print(e, 'Training failed')
                raise

        avg_loss = total_loss / self.len_data_loader
        avg_det_loss = total_det_loss / self.len_data_loader
        avg_rec_loss = total_rec_loss / self.len_data_loader
        total_text_accuracy = text_accuracy / self.len_data_loader
        total_text_accuracy_wo_decimal = text_accuracy_wo_decimal / self.len_data_loader
        total_char_similarity = char_similarity / self.len_data_loader
        total_value_mae = value_mae / self.len_data_loader
        total_time = round(time.time() - start_time, 2)
        if self.verbosity >= 2:
            self.logger.info('Train: Epoch: {} [{} samples, {} seconds] \
Loss: {:.2f} \
Det Loss: {:.2f} \
Rec Loss: {:.2f} \
Char similarity {:.2f}'.format(
                    epoch,
                    len(self.data_loader.dataset),
                    total_time,
                    avg_loss, avg_det_loss, avg_rec_loss, total_char_similarity))

        log = {
            'loss': avg_loss,
            'det_loss': avg_det_loss,
            'rec_loss': avg_rec_loss,
            'acc': total_text_accuracy,
            'acc_wo_decimal': total_text_accuracy_wo_decimal,
            'char_similarity': total_char_similarity,
            'value_mae': total_value_mae,
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
        text_accuracy, text_accuracy_wo_decimal, char_similarity, value_mae = 0.0, 0.0, 0.0, 0.0
        start_time = time.time()
        with torch.no_grad():
            for batch_idx, gt in enumerate(self.valid_data_loader):
                try:
                    image_files, _image, score_map, transcriptions, boxes = gt
                    image, score_map = self._to_device(_image.clone(), score_map.clone())

                    pred_score_map, pred_recog, pred_boxes, indices = self.model.forward(image_files, image, boxes)
                    gt_transcriptions = transcriptions[indices]
                    image_files = np.array(image_files)[indices]
                    image_visual = _image[indices]
                    pred_boxes = pred_boxes[indices]
                    pred_score_map = pred_score_map[indices]
                    score_map = score_map[indices]

                    labels, label_lengths = self.label_converter.encode(gt_transcriptions.tolist())
                    labels = labels.to(self.device)
                    label_lengths = label_lengths.to(self.device)
                    recog = (labels, label_lengths)

                    ## Get transcription predictions
                    pred_transcriptions = []
                    pred, lengths = pred_recog
                    _, pred = pred.max(2)
                    for i in range(lengths.numel()):
                        l = lengths[i]
                        p = pred[:l, i]
                        t = self.label_converter.decode(p, l)
                        pred_transcriptions.append(t)
                    pred_transcriptions = np.array(pred_transcriptions)

                    if batch_idx == 0:
                        print('Validation gt transcriptions:', gt_transcriptions[:8])
                        print('Validation pred transcriptions:', pred_transcriptions[:8])

                    ## Write summary writer images and text
                    if epoch % self.save_freq == 0 and batch_idx == 0:
                        step = epoch * self.len_valid_data_loader + batch_idx

                        image_grid = torchvision.utils.make_grid(image_visual, normalize=True, scale_each=True)
                        score_map_grid = torchvision.utils.make_grid(score_map)
                        pred_score_map_grid = torchvision.utils.make_grid(pred_score_map.double())

                        gt_transcriptions_tensor = draw_text_tensor(gt_transcriptions)
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

                    ## Get loss
                    detection_loss, recognition_loss = self.loss(pred_score_map, score_map, pred_recog, recog)
                    loss = detection_loss + recognition_loss
                    total_loss += loss.item()
                    total_det_loss += detection_loss.item()
                    total_rec_loss += recognition_loss.item()

                    ## Text accuracy
                    text_accuracy += np.mean(gt_transcriptions == pred_transcriptions)

                    ## Text without decimal part accuracy
                    gt_trans_wo_decimal = np.array(list(map(lambda s: s.split(',')[0], gt_transcriptions))).astype(str)
                    pred_trans_wo_decimal = np.array(list(map(lambda s: s.split(',')[0], pred_transcriptions))).astype(str)
                    text_accuracy_wo_decimal += np.mean(gt_trans_wo_decimal == pred_trans_wo_decimal)

                    ## Char accuracy by SequenceMatcher
                    char_similarity += get_mean_char_similarity(pred_transcriptions, gt_transcriptions)

                    ## Value MSE
                    gt_transcription_values = np.array(list(map(self.get_transcription_value, gt_transcriptions))).astype(np.float32)
                    pred_transcription_values = np.array(list(map(self.get_transcription_value, pred_transcriptions))).astype(np.float32)
                    value_mae += mae(gt_transcription_values, pred_transcription_values)

                except Exception as e:
                    print(e, 'Validation failed')
                    # raise

        avg_loss = total_loss / self.len_valid_data_loader
        avg_det_loss = total_det_loss / self.len_valid_data_loader
        avg_rec_loss = total_rec_loss / self.len_valid_data_loader
        total_text_accuracy = text_accuracy / self.len_valid_data_loader
        total_text_accuracy_wo_decimal = text_accuracy_wo_decimal / self.len_valid_data_loader
        total_char_similarity = char_similarity / self.len_valid_data_loader
        total_value_mae = value_mae / self.len_valid_data_loader
        total_time = round(time.time() - start_time, 2)
        if self.verbosity >= 2:
            self.logger.info('Val: Epoch: {} [{} samples, {:.2f} seconds] \
Loss: {:.4f} \
Det Loss: {:.4f} \
Rec Loss: {:.4f} \
Char similarity {:.2f}'.format(
                    epoch,
                    len(self.valid_data_loader.dataset),
                    total_time,
                    avg_loss, avg_det_loss, avg_rec_loss, total_char_similarity))

        return {
            'val_loss': avg_loss,
            'val_det_loss': avg_det_loss,
            'val_rec_loss': avg_rec_loss,
            'val_acc': total_text_accuracy,
            'val_acc_wo_decimal': total_text_accuracy_wo_decimal,
            'val_char_similarity': total_char_similarity,
            'val_value_mae': total_value_mae,
            'val_time': total_time
        }

    def get_transcription_value(self, text, delimiter=','):
        text = text.strip()
        if text == '':
            return 0
        if delimiter in text:
            whole, decimal = text.split(delimiter)[:2]
        else:
            whole = int(text)
            decimal = 0
        try:
            return int(whole) + int(decimal) / 100.0
        except ValueError:
            return 0
