import json
import logging
import math
import os
import shutil

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from ..utils.util import ensure_dir

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, resume, config, config_file, train_logger=None):
        self.config = config
        self.config_file = config_file
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.name = config['name']
        self.epochs = config['trainer']['epochs']
        self.save_freq = config['trainer']['save_freq']
        self.verbosity = config['trainer']['verbosity']
        self.summary_writer = SummaryWriter()

        ## Copy config file to summary writer folder ('runs/')
        shutil.copy(self.config_file, self.summary_writer.file_writer.get_logdir())

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
                                'training is performed on CPU.')
            self.with_cuda = False
            device = 'cpu'

        self.device = torch.device(device)
        self.model.to(self.device)

        self.logger.debug('Model is initialized.')
        self._log_memory_usage()

        self.train_logger = train_logger

        self.optimizer = self.model.optimize(config['optimizer_type'], config['optimizer'])

        self.lr_scheduler = getattr(
            optim.lr_scheduler,
            config['lr_scheduler_type'], None)
        if self.lr_scheduler:
            self.lr_scheduler = self.lr_scheduler(self.optimizer, **config['lr_scheduler'])
        self.monitor = config['trainer']['monitor']
        self.monitor_mode = config['trainer']['monitor_mode']
        assert self.monitor_mode == 'min' or self.monitor_mode == 'max'
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
        self.start_epoch = 1
        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], self.name)
        ensure_dir(self.checkpoint_dir)
        json.dump(config, open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4, sort_keys=False)
        if resume:
            self._resume_checkpoint(resume)

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            try:
                result = self._train_epoch(epoch)
            except torch.cuda.CudaError:
                self._log_memory_usage()

            log = {'epoch': epoch}
            for key, value in result.items():
                log[key] = value

            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        if str(key) == 'epoch':
                            self.logger.info('\t{:20s}: {}'.format(str(key), value))
                        else:
                            self.logger.info('\t{:20s}: {:.4f}'.format(str(key), value))
            if (self.monitor_mode == 'min' and log[self.monitor] < self.monitor_best)\
                    or (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
                self.monitor_best = log[self.monitor]
                self._save_checkpoint(epoch, log, save_best=True)
            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch, log)
            if self.lr_scheduler:
                self.lr_scheduler.step(log[self.monitor])
                lr = self.lr_scheduler.optimizer.param_groups[0]['lr']

            self.summary_writer.add_scalar('Train_loss', result['loss'], epoch)
            self.summary_writer.add_scalar('Train_detection_loss', result['det_loss'], epoch)
            self.summary_writer.add_scalar('Train_recognition_loss', result['rec_loss'], epoch)
            self.summary_writer.add_scalar('Train_text_accuracy', result['acc'], epoch)
            self.summary_writer.add_scalar('Train_text_accuracy_wo_decimal', result['acc_wo_decimal'], epoch)
            self.summary_writer.add_scalar('Train_char_similarity', result['char_similarity'], epoch)
            self.summary_writer.add_scalar('Train_value_mae', result['value_mae'], epoch)
            self.summary_writer.add_scalar('Train_time', result['time'], epoch)
            self.summary_writer.add_scalar('Val_loss', result['val_loss'], epoch)
            self.summary_writer.add_scalar('Val_detection_loss', result['val_det_loss'], epoch)
            self.summary_writer.add_scalar('Val_recognition_loss', result['val_rec_loss'], epoch)
            self.summary_writer.add_scalar('Val_text_accuracy', result['val_acc'], epoch)
            self.summary_writer.add_scalar('Val_text_accuracy_wo_decimal', result['val_acc_wo_decimal'], epoch)
            self.summary_writer.add_scalar('Val_char_similarity', result['val_char_similarity'], epoch)
            self.summary_writer.add_scalar('Val_value_mae', result['val_value_mae'], epoch)
            self.summary_writer.add_scalar('Val_time', result['val_time'], epoch)
            self.summary_writer.add_scalar('Learning_rate', lr, epoch)
        self.summary_writer.close()

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

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        """
        Validation logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, log, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'config': self.config,
            'epoch': epoch,
            'logger': self.train_logger,
            'monitor_best': self.monitor_best,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict()
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{:03d}-loss-{:.4f}.pth.tar'
                                .format(epoch, log['loss']))
        torch.save(state, filename)
        if save_best:
            os.rename(filename, os.path.join(self.checkpoint_dir, 'model_best.pth.tar'))
            self.logger.info("Saving current best: {}".format('model_best.pth.tar'))
        else:
            self.logger.info("Saving checkpoint: {}".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {}".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.with_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(torch.device('cuda'))
        self.train_logger = checkpoint['logger']
        #self.config = checkpoint['config']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
