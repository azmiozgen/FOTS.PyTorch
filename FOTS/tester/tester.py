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
from ..model.metric import get_mean_char_similarity, mae
from ..utils.util import strLabelConverter

class Tester(BaseTester):
    def __init__(self, model, model_file, metrics, config, data_loader):
        super(Tester, self).__init__(model, model_file, metrics, config)
        self.config = config
        self.batch_size = config['tester']['batch_size']
        self.data_loader = data_loader
        self.epoch_record = config['tester']['epoch_record']
        self.label_converter = strLabelConverter(keys)
        self.len_data_loader = len(data_loader)
        self.dataset_size = len(data_loader.dataset)
        self.output_dir = config['tester']['save_dir']
        os.makedirs(self.output_dir, exist_ok=True)

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
        self.model_name = os.path.basename(os.path.dirname(model_file))
        self.model_filename, self.model_file_ext = os.path.splitext(os.path.basename(model_file))
        self.logger.info(f'Model {model_file} loaded')

    def test(self):
        self.model.eval()
        text_accuracy, text_accuracy_wo_decimal, char_similarity, value_mae = 0.0, 0.0, 0.0, 0.0
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
                        print('Test sample gt transcriptions:', gt_transcriptions[:8])
                        print('Test sample pred transcriptions:', pred_transcriptions[:8])

                        image_grid = torchvision.utils.make_grid(image_visual, normalize=True, scale_each=True)
                        gt_score_map_grid = torchvision.utils.make_grid(score_map)
                        pred_score_map_grid = torchvision.utils.make_grid(pred_score_map.double())

                        gt_transcriptions_tensor = draw_text_tensor(gt_transcriptions)
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
                    value_mae += mae(gt_transcription_values, pred_transcription_values)

                except Exception as e:
                    print(e, 'Testing failed')
                    raise

        ## Make grid images proper numpy array
        grids = [gt_score_map_grid, image_grid, gt_transcriptions_grid, pred_score_map_grid, pred_transcriptions_grid]
        grid_titles = ['GT Masks', 'Images', 'GT Transcriptions', 'Pred Masks', 'Pred Transcriptions']
        grids = [grid.cpu().numpy().transpose(1, 2, 0) for grid in grids]

        total_text_accuracy = text_accuracy / self.len_data_loader
        total_text_accuracy_wo_decimal = text_accuracy_wo_decimal / self.len_data_loader
        total_char_similarity = char_similarity / self.len_data_loader
        total_value_mae = value_mae / self.len_data_loader
        total_time = round(time.time() - start_time, 2)
        self.logger.info('Test: [{} samples, {:.2f} seconds]\
\n\tText accuracy: {:.2f}\
\n\tText accuracy without decimal: {:.2f}\
\n\tCharacter similarity: {:.2f}\
\n\tValue MAE: {:.2f}'.format(
                len(self.data_loader.dataset),
                total_time,
                total_text_accuracy,
                total_text_accuracy_wo_decimal,
                total_char_similarity,
                total_value_mae))

        result = {
            'size': self.dataset_size,
            'text_accuracy': round(total_text_accuracy, 2),
            'text_accuracy_wo_decimal': round(total_text_accuracy_wo_decimal, 2),
            'char_similarity': round(total_char_similarity, 2),
            'value_mae': round(total_value_mae, 2),
        }

        self.visualize(result, grids, grid_titles)

    def visualize(self, result, grids, grid_titles,
                fig_size=(15, 5),
                fig_style = 'ggplot',
                fig_subplot_size = (2, 3),
                fontsize=16):
        assert len(grids) == len(grid_titles) == 5
        self.logger.info('Visualizing results..')
        n_cols=3
        n_rows=2

        title = self.model_name + f'_epoch{self.epoch_record}'
        output_filename = title + '_test_result.png'
        output_file = os.path.join(self.output_dir, output_filename)

        plt.style.use(fig_style)
        fig, axes = plt.subplots(*fig_subplot_size, figsize=fig_size)
        fig.tight_layout()
        fig.suptitle(title, fontsize=fontsize)

        ## Put images
        for i in range(n_rows):
            for j in range(n_cols):
                if i == 1 and j == 2:
                    continue
                axes[i, j].imshow(grids[i * n_cols + j])
                axes[i, j].set_title(grid_titles[i * n_cols + j], fontsize=fontsize)
                axes[i, j].axis('off')

        ## Put accuracy table
        table_keys = ['N Samples', 'Text accuracy', 'Text accuracy\nwithout decimal', 'Character similarity', 'Value MAE']
        table_vals = np.array(list(result.values())).reshape(-1, 1)
        axes[1, 2].axis('off')
        axes[1, 2].table(cellText=table_vals,
                rowLabels=table_keys,
                cellLoc='left',
                bbox=[0.4, 0, 0.2, 0.75],
                colWidths=[0.2, 4]
                )

        fig.savefig(output_file, dpi=300)
        self.logger.info(f'{output_file} was written.')
        plt.close(fig)

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