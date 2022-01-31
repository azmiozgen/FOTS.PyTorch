import argparse
import json
import logging
import os
import pathlib

from numpy import require

from FOTS.data_loader import ICDAR
from FOTS.logger import Logger
from FOTS.model.model import *
from FOTS.model.loss import *
from FOTS.model.metric import *
from FOTS.trainer import Trainer
from FOTS.utils.bbox import Toolbox

logging.basicConfig(level=logging.INFO, format='')


def main(config, resume):
    train_logger = Logger()

    if config['data_loader']['dataset'] == 'icdar2015':
        from FOTS.data_loader import OCRDataLoaderFactory
        # ICDAR 2015
        data_root = pathlib.Path(config['data_loader']['data_dir'])
        ICDARDataset2015 = ICDAR(data_root, year='2015')
        data_loader = OCRDataLoaderFactory(config, ICDARDataset2015)
        train = data_loader.train()
        val = data_loader.val()
    elif config['data_loader']['dataset'] == 'synth800k':
        from FOTS.data_loader import SynthTextDataLoaderFactory
        data_loader = SynthTextDataLoaderFactory(config)
        train = data_loader.train()
        val = data_loader.val()
    elif config['data_loader']['dataset'] == 'json':
        from FOTS.data_loader import PriceTagDataLoaderFactory
        data_loader = PriceTagDataLoaderFactory(config)
        train = data_loader.train()
        val = data_loader.val()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config['gpus']])
    model = eval(config['arch'])(config)
    # model.summary()

    print('Training size:', len(data_loader.train_dataset))
    print('Validation size:', len(data_loader.val_dataset))

    # for batch_i, gt in enumerate(train):
    #     image_files, images, score_maps, geo_maps, training_masks, transcriptions, boxes, mapping = gt
    #     print('image_files:', image_files)
    #     print('images', images.shape)
    #     print('transcriptions', transcriptions)
    #     print('boxes', boxes)
    #     print('score_maps', score_maps.shape)
    #     print('geo_maps', geo_maps.shape)
    #     print('training_masks', training_masks.shape)
    #     print('mapping', mapping)
    #     # transformed_image_filename = image_filename_wo_ext + '_transformed.' + ext
    #     # transformed_image_file = os.path.join(self.visualization_dir, transformed_image_filename)
    #     # cv2.imwrite(transformed_image_file, cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
    #     # self.visualize(transformed_image_file, rectangles[0], transcriptions[0])
    #     break

    # for batch_i, gt in enumerate(val):
    #     print(batch_i, gt)
    #     break

    loss = eval(config['loss'])(config)
    metrics = [eval(metric) for metric in config['metrics']]  ## precision, recall, hmean (f1)

    trainer = Trainer(model, loss, metrics,
                      resume=resume,
                      config=config,
                      data_loader=train,
                      valid_data_loader=val,
                      train_logger=train_logger,
                      toolbox=Toolbox)

    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='config file path (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    config = None
    if args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        #assert not os.path.exists(path), "Path {} already exists!".format(path)
    else:
        if args.resume is not None:
            logger.warning('Warning: --config overridden by --resume')
            config = torch.load(args.resume, map_location='cpu')['config']

    assert config is not None

    main(config, args.resume)
