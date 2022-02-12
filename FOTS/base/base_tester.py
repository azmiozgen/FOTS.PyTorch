import logging
import os

import torch


class BaseTester:
    """
    Base class for all testers
    """
    def __init__(self, model, model_file, metrics, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.metrics = metrics
        self.name = config['name']
        self.model_filename = None
        self.model_file_ext = None

        if torch.cuda.is_available():
            if config['cuda']:
                self.with_cuda = True
                self.gpus = {i: item for i, item in enumerate(self.config['gpus'])}
                device = 'cuda'
                if torch.cuda.device_count() > 1 and len(self.gpus) > 1:
                    self.model.parallelize()
                torch.cuda.empty_cache()
            else:
                self.with_cuda = False
                device = 'cpu'
        else:
            self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'testing is performed on CPU.')
            self.with_cuda = False
            device = 'cpu'

        self.device = torch.device(device)
        self.model.to(self.device)

        self.logger.debug('Model is initialized.')
        self._log_memory_usage()

        self._load_model(model_file)

    def _load_model(self, model_file):
        self.logger.info("Loading model: {} ...".format(model_file))
        model = torch.load(model_file)
        self.model.load_state_dict(model['state_dict'])
        self.model_name = os.path.basename(os.path.dirname(model_file))
        self.model_filename, self.model_file_ext = os.path.splitext(os.path.basename(model_file))

    def _log_memory_usage(self):
        if not self.with_cuda: return

        template = """Memory Usage: \n{}"""
        usage = []
        for deviceID, device in self.gpus.items():
            deviceID = int(deviceID)
            allocated = torch.cuda.memory_allocated(deviceID) / (1024 * 1024)
            cached = torch.cuda.memory_cached(deviceID) / (1024 * 1024)

            usage.append('    CUDA: {}  Allocated: {} MB Cached: {} MB \n'.format(device, allocated, cached))

        content = ''.join(usage)
        content = template.format(content)

        self.logger.debug(content)

    def test(self):
        raise NotImplementedError