import cv2
import glob
import json
import logging
import os
import sys

from FOTS.model.model import FOTSModel
from FOTS.tester import Predictor

logging.basicConfig(level=logging.INFO, format='')

PREDICTION_CONFIG = {
    'arch': 'FOTSModel',
    'batch_size': 16,
    'cuda': True,
    'gpus': [0],
    'input_size': 256,
    'num_workers': 4,
}
MODEL_FILE = os.path.join('runs', 'Feb11_12-41-31_kowalski', 'model_best_epoch050.pth.tar')
MODEL_CONFIG_FILE = os.path.join('runs', 'Feb11_12-41-31_kowalski', 'config.json')

def main(model_config, prediction_config, model_file, images):

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in prediction_config['gpus']])

    ## Get model architecture
    model_name = model_config['arch']
    if model_name == 'FOTSModel':
        model = FOTSModel(model_config)
    else:
        raise NotImplementedError(f'{model_name} not implemented')

    predictor = Predictor(model, model_file, prediction_config, images)

    ## Print info
    print('Model:', model_config['arch'])
    print('Sample size:', predictor.dataset_size)

    predictor.predict()

def read_images(image_files):
    return [cv2.imread(image_file) for image_file in image_files]

if __name__ == '__main__':
    ## Get image file
    if len(sys.argv) != 2:
        print(f'Usage:  python {__file__} <images_dir>')
        sys.exit()
    images_dir = sys.argv[1]

    if not os.path.isdir(images_dir):
        print(f'{images_dir} was not found. Exiting.')
        sys.exit()

    if not os.path.isfile(MODEL_FILE):
        print(f'{MODEL_FILE} was not found. Exiting.')
        sys.exit()

    if not os.path.isfile(MODEL_CONFIG_FILE):
        print(f'{MODEL_CONFIG_FILE} was not found. Exiting.')
        sys.exit()
    with open(MODEL_CONFIG_FILE, 'r') as f:
        model_config = json.load(f)

    image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
    print(f'Found {len(image_files)} images')
    images = read_images(image_files)

    main(model_config, PREDICTION_CONFIG, MODEL_FILE, images)
