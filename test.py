import argparse
import json
import logging
import os

from FOTS.model.model import FOTSModel
from FOTS.tester import Tester

logging.basicConfig(level=logging.INFO, format='')


def main(config_file, model_file):

    ## Load config and model file
    assert os.path.isfile(config_file), f'{config_file} config not found'
    with open(config_file, 'r') as f:
        config = json.load(f)

    if config['data_loader']['dataset'] == 'price_tag':
        from FOTS.data_loader import PriceTagDataLoaderFactory
        data_loader = PriceTagDataLoaderFactory(config)
        test = data_loader.test()

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
    print('Test size:', len(data_loader.test_dataset))
    print('Test batch size:', config['tester']['batch_size'])

    metrics = None ## TODO: add metrics

    tester = Tester(model, model_file, metrics,
            config=config,
            data_loader=test)

    tester.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-m', '--model', default=None, type=str, required=True,
                        help='Model file (default: None)')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='config file path (default: config.json)')

    args = parser.parse_args()
    config_file = args.config
    model_file = args.model

    main(config_file, model_file)
