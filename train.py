import argparse
import json
import logging
import os

from FOTS.logger import Logger
from FOTS.model.model import *
from FOTS.model.loss import *
from FOTS.model.metric import *
from FOTS.trainer import Trainer
from FOTS.utils.bbox import Toolbox

logging.basicConfig(level=logging.INFO, format='')


def main(config_file, resume):
    train_logger = Logger()

    ## Load config file and resume checkpoint
    assert os.path.isfile(config_file), f'{config_file} config not found'
    with open(config_file, 'r') as f:
        config = json.load(f)

    if config['data_loader']['dataset'] == 'price_tag':
        from FOTS.data_loader import PriceTagDataLoaderFactory
        data_loader = PriceTagDataLoaderFactory(config)
        train = data_loader.train()
        val = data_loader.val()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config['gpus']])
    model = eval(config['arch'])(config)
    # model.summary()

    ## Print info
    print('Model:', config['arch'])
    print('Mode:', config['model']['mode'])
    print('Training size:', len(data_loader.train_dataset))
    print('Validation size:', len(data_loader.val_dataset))

    loss = eval(config['loss'])(config)
    metrics = [eval(metric) for metric in config['metrics']]  ## precision, recall, hmean (f1)

    trainer = Trainer(model, loss, metrics,
            resume=resume,
            config=config,
            config_file=config_file,
            data_loader=train,
            valid_data_loader=val,
            train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='config file path (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()
    config_file = args.config
    resume = args.resume

    main(config_file, resume)
