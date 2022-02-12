from collections import OrderedDict
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from ..base import BaseTester
from ..data_loader.datautils import draw_text_tensor
from ..model.keys import keys
from ..model.metric import rmse, get_mean_char_similarity
from ..utils.util import strLabelConverter

class Tester(BaseTester):
    def __init__(self, model, model_file, metrics, config, data_loader):
        super(Tester, self).__init__(model, model_file, metrics, config)
        self.config = config
        self.batch_size = config['tester']['batch_size']
        self.data_loader = data_loader
        self.label_converter = strLabelConverter(keys)
        self.len_data_loader = len(data_loader)
        self.dataset_size = len(data_loader.dataset)
        self.output_dir = config['tester']['save_dir']

    def _to_device(self, *tensors):
        t = []
        for __tensors in tensors:
            t.append(__tensors.to(self.device))
        return t

    def _load_model(self, model_file):
        self.logger.info("Loading model: {} ...".format(model_file))
        model = torch.load(model_file, map_location=self.device)
        try:
            self.model.load_state_dict(model['state_dict'])
        except RuntimeError:  ## Load nn.DataParallel
            for s in ['0', '1', '2']:  ## SharedConv, Detector, Recognizer
                new_state_dict = OrderedDict()
                for k, v in model['state_dict'][s].items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                model['state_dict'][s] = new_state_dict
            self.model.load_state_dict(model['state_dict'])
        self.model_filename, self.model_file_ext = os.path.splitext(os.path.basename(model_file))
        self.logger.info(f'Model {model_file} loaded')

    def test(self):
        self.model.eval()
        text_accuracy, text_accuracy_wo_decimal, char_similarity, value_rmse  = 0.0, 0.0, 0.0, 0.0
        start_time = time.time()
        with torch.no_grad():
            for batch_idx, gt in enumerate(self.data_loader):
                try:
                    image_files, _image, score_map, transcriptions, boxes = gt
                    image, score_map = self._to_device(_image.clone(), score_map.clone())

                    pred_score_map, pred_recog, pred_boxes, indices = self.model.forward(image_files, image, boxes)
                    gt_transcriptions = np.array(transcriptions[indices]).astype(str)
                    image_files = np.array(image_files)[indices]
                    image_visual = _image[indices]
                    pred_boxes = pred_boxes[indices]
                    pred_score_map = pred_score_map[indices]
                    score_map = score_map[indices]

                    ## Get transcription predictions
                    pred_transcriptions = []
                    pred, lengths = pred_recog
                    _, pred = pred.max(2)
                    for i in range(lengths.numel()):
                        l = lengths[i]
                        p = pred[:l, i]
                        t = self.label_converter.decode(p, l)
                        pred_transcriptions.append(t)
                    pred_transcriptions = np.array(pred_transcriptions).astype(str)

                    if batch_idx == 0:
                        print('Test sample gt transcriptions:', transcriptions[:8])
                        print('Test sample pred transcriptions:', pred_transcriptions[:8])

                        image_grid = torchvision.utils.make_grid(image_visual, normalize=True, scale_each=True)
                        gt_score_map_grid = torchvision.utils.make_grid(score_map)
                        pred_score_map_grid = torchvision.utils.make_grid(pred_score_map.double())

                        gt_transcriptions_tensor = draw_text_tensor(transcriptions)
                        gt_transcriptions_grid = torchvision.utils.make_grid(gt_transcriptions_tensor,
                                normalize=True, value_range=(0, 1))

                        pred_transcriptions_tensor = draw_text_tensor(pred_transcriptions)
                        pred_transcriptions_grid = torchvision.utils.make_grid(pred_transcriptions_tensor,
                                normalize=True, value_range=(0, 1))

                    ## Text accuracy
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
                    value_rmse += rmse(pred_transcription_values, gt_transcription_values)

                except Exception as e:
                    print(e, 'Testing failed')
                    raise

        grids = [image_grid, gt_score_map_grid, pred_score_map_grid, gt_transcriptions_grid, pred_transcriptions_grid]
        total_text_accuracy = text_accuracy / self.len_data_loader
        total_text_accuracy_wo_decimal = text_accuracy_wo_decimal / self.len_data_loader
        total_char_similarity = char_similarity / self.len_data_loader
        total_value_rmse = value_rmse / self.len_data_loader
        total_time = round(time.time() - start_time, 2)
        self.logger.info('Test: [{} samples, {:.2f} seconds]\
\n\tText accuracy: {:.2f}\
\n\tText accuracy without decimal: {:.2f}\
\n\tCharacter similarity: {:.2f}\
\n\tValue RMSE: {:.2f}'.format(
                len(self.data_loader.dataset),
                total_time,
                total_text_accuracy,
                total_text_accuracy_wo_decimal,
                total_char_similarity,
                total_value_rmse))

        log = {
            'size': self.dataset_size,
            'text_accuracy': total_text_accuracy,
            'text_accuracy_wo_decimal': total_text_accuracy_wo_decimal,
        }

        # self.visualize(log, grids)

    def visualize(self, log, grids,
                figsize=(10, 10),
                fontsize=16,
                scale=1.0):
        log.info('Visualizing...')
        output_filename = self.model_filename + '_test_result.png'
        output_file = os.path.join(self.output_dir, output_filename)

        ## Create matplotlib table image for text accuracy
        _, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        ax.axis('tight')
        table_data = log
        table = ax.table(cellText=table_data, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.scale(scale, scale)
        plt.savefig(output_file, dpi=300)
        plt.close()

    def get_transcription_value(self, text, delimiter=','):
        if delimiter in text:
            whole, decimal = text.split(delimiter)
        else:
            whole = int(text)
            decimal = 0
        try:
            return int(whole) + int(decimal) / 100.0
        except ValueError:
            return 0