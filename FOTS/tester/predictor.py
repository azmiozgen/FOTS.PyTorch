from collections import OrderedDict
import logging
import os
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.utils.data as torchdata

from ..data_loader.dataset import PriceTagPredictionDataset
from ..data_loader.datautils import collate_images
from ..model.keys import keys
from ..utils.util import strLabelConverter


class Predictor:
    def __init__(self, model, model_file, config, images):
        self.label_converter = strLabelConverter(keys)
        self.model = model
        self.model_file = model_file
        self.config = config
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.images = images
        self.logger = logging.getLogger(self.__class__.__name__)

        ## Load dataset
        self.dataset = PriceTagPredictionDataset(images, input_size=config['input_size'])
        self.dataset_size = len(self.dataset)
        self.data_loader = None
        self._load_dataset()

        ## Set device
        if torch.cuda.is_available():
            if config['cuda']:
                self.gpus = {i: item for i, item in enumerate(self.config['gpus'])}
                device = 'cuda'
                if torch.cuda.device_count() > 1 and len(self.gpus) > 1:
                    self.model.parallelize()
                torch.cuda.empty_cache()
            else:
                device = 'cpu'
        else:
            self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'prediction is performed on CPU.')
            device = 'cpu'
        self.device = torch.device(device)

        ## Load model weights
        self._load_model(model_file)
        self.model.to(self.device)
        self.logger.debug('Model is initialized.')

    def _load_dataset(self):
        self.data_loader = torchdata.DataLoader(self.dataset,
                        num_workers=self.num_workers,
                        batch_size=min(self.dataset_size, self.batch_size),
                        shuffle=False,
                        collate_fn=collate_images)

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

    def predict(self):
        self.model.eval()
        start_time = time.time()
        predictions = []
        with torch.no_grad():
            for images in tqdm(self.data_loader):
                try:
                    ## Forward pass
                    images = images.to(self.device)
                    _, pred_recog, _, indices = self.model.forward([], images, [])

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
                    pred_transcriptions = pred_transcriptions[np.argsort(indices)]  ## Revert the order of predictions
                    predictions.extend(pred_transcriptions)

                except Exception as e:
                    print(e, 'Testing failed')
                    raise

        total_time = round(time.time() - start_time, 2)

        self.logger.info('Prediction: [{} samples, {:.2f} seconds]'.format(
                self.dataset_size,
                total_time))

        for prediction in predictions:
            self.logger.info('\t' + prediction)

        result = {
            'predictions': predictions,
            'size': self.dataset_size,
            'time': total_time,
        }

        return result