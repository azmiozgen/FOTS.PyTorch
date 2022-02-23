import argparse
import json
import os

import torch
from torch.onnx import export

from FOTS.model.model import FOTSModel


def main(config_file, output_file):

    ## Load config
    assert os.path.isfile(config_file), f'{config_file} config not found'
    with open(config_file, 'r') as f:
        config = json.load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config['gpus']])

    ## Get model architecture
    model_name = config['arch']
    if model_name == 'FOTSModel':
        model = FOTSModel(config)
        model.eval()
    else:
        raise NotImplementedError(f'{model_name} not implemented')
    # model.summary()

    ## Print info
    print('Model:', config['arch'])

    # batch_size = config['data_loader']['batch_size']
    batch_size = 1
    input_size = config['data_loader']['input_size']
    x_shared_conv = torch.randn(batch_size, 3, input_size, input_size, requires_grad=True)
    # x_detector = torch.randn(batch_size, 32, input_size // 4, input_size // 4, requires_grad=True)
    x_recognizer = torch.randn(batch_size, 32, input_size // 4, input_size // 4, requires_grad=True)
    lengths = torch.tensor([input_size // 4], dtype=torch.int64)
    _ = model.forward([], x_shared_conv, [])
    export(model.recognizer, (x_recognizer, lengths), output_file, export_params=False, opset_version=11)
    print(output_file, 'written.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='config file path (default: config.json)')
    parser.add_argument('-o', '--output', default='model.onnx', type=str,
                        help='Output onnx file (default: model.onnx)')

    args = parser.parse_args()
    config_file = args.config
    output_file = args.output

    main(config_file, output_file)
