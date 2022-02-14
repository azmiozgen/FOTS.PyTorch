import argparse
import json
import logging
import os

from FOTS.logger.logger import Logger
from FOTS.model.model import FOTSModel
from FOTS.model.loss import FOTSLoss
from FOTS.trainer.trainer import Trainer

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

    ## Get model architecture
    model_name = config['arch']
    if model_name == 'FOTSModel':
        model = FOTSModel(config)
    else:
        raise NotImplementedError(f'{model_name} not implemented')
    # model.summary()

    ## Print info
    print('Model:', config['arch'])
    print('Mode:', config['model']['mode'])
    print('Batch size:', config['data_loader']['batch_size'])
    print('Learning rate:', config['optimizer']['lr'])
    print('Training size:', len(data_loader.train_dataset))
    print('Validation size:', len(data_loader.val_dataset))

    ## Get loss
    loss_name = config['loss']
    if loss_name == 'FOTSLoss':
        loss = FOTSLoss(config)
    else:
        raise NotImplementedError(f'{loss_name} not implemented')

    ## Get metrics
    metrics = None

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
