import logging
from datetime import datetime

from plotting import *
from utils.totext import as_giga_mega_kilo_bytes

from typing import Optional, Tuple, BinaryIO, Dict
import torch.cuda as cuda

import os.path as path


class Manager(object):
    def __init__(self):
        self._loss = 0
        self.losses = []
        self._accuracy = 0
        self.accuracies = []

        self.time_elapsed, self.time_stop, self.time_start = None, None, None

    def __enter__(self):
        self.time_start = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time_stop = datetime.now()
        self.time_elapsed = self.time_stop - self.time_start

    def loss(self) -> float:
        if len(self.losses) > 0:
            self._loss = sum(self.losses) / len(self.losses)

        return self._loss

    def accuracy(self) -> float:
        if len(self.accuracies) > 0:
            self._accuracy = sum(self.accuracies) / len(self.accuracies)

        return self._accuracy

    def add(self, loss: float, accuracy: float):
        self.losses.append(loss)
        self.accuracies.append(accuracy)

    def clear(self):
        self.losses.clear()


class TrackerArgs:
    """Arguments tu build a Tracker object"""

    def __init__(self,
                 save_folder: str,
                 data_buffer_size: Optional[int] = 1):
        """

        :param save_folder: folder in which save data is written
        :param data_buffer_size: size of the buffer used to save data, when this buffer is full the data is written to a
        file
        """
        self.save_folder = save_folder
        self.data_buffer_size = data_buffer_size


class Tracker:
    def __init__(self,
                 parent_trainer: object,
                 tracker_args: TrackerArgs):
        """Create a new tracking tool for data such as time, memory and accuracy over epochs

        :param parent_trainer: parent trainer
        """
        self.trainer = parent_trainer

        # Data storage
        self.save_folder = tracker_args.save_folder
        self.sequence_data = DataFile(path.join(self.save_folder, FILE_SEQUENCE),
                                      keys=[EPOCH, TIME_VALID, TIME_SEQ, LOSS, LOSS_VALID, ACCURACY, ACCURACY_VALID,
                                            BATCH_SIZE],
                                      buffer_size=tracker_args.data_buffer_size)
        self.memory_data = DataFile(path.join(self.save_folder, FILE_MEMORY),
                                    keys=[EPOCH, CUDA_MEM_AVAIL, CUDA_MEM_USE],
                                    buffer_size=tracker_args.data_buffer_size)
        self.epoch_data = DataFile(path.join(self.save_folder, FILE_EPOCH),
                                   keys=[EPOCH, TIME, LOSS, LOSS_VALID, ACCURACY, ACCURACY_VALID])

        # Clear data storage if this is not a loaded tracker
        self.sequence_data.clear_file()
        self.memory_data.clear_file()
        self.epoch_data.clear_file()

        # Managers (time and data managers)
        self.manager_test = Manager()
        self.manager_validation = Manager()
        self.manager_sequence = Manager()
        self.manager_epoch = Manager()  # Add interruption support

    def test(self) -> Manager:
        """Get a validation manager"""
        return self.manager_validation

    def validation(self) -> Manager:
        """Get a validation manager"""
        return self.manager_validation

    def sequence(self) -> Manager:
        """Get a sequence manager"""

        return self.manager_sequence

    def epoch(self) -> Manager:
        """Get an epoch manager"""

        return self.manager_epoch

    def log_test(self) -> Tuple[float, float, float]:
        # Fetch losses and clear buffers
        accuracy, loss = self.manager_test.accuracy(), self.manager_test.loss()
        self.manager_test.clear()

        data = {
            EPOCH: self._epoch(),
            LOSS: loss,
            ACCURACY: accuracy * 100
        }
        logging.warning("Test log: {}".format(data))
        return data[EPOCH], data[LOSS], data[ACCURACY]

    def log_sequence(self, status: Dict[str, str]):
        # Fetch losses and clear buffers
        loss, loss_valid = self.manager_sequence.loss(), self.manager_validation.loss()
        accuracy, accuracy_valid = self.manager_sequence.accuracy(), self.manager_validation.accuracy()
        self.manager_sequence.clear()
        self.manager_validation.clear()

        data = {
            EPOCH: self._epoch(status),
            TIME_SEQ: self.manager_sequence.time_elapsed.total_seconds(),
            TIME_VALID: self.manager_validation.time_elapsed.total_seconds(),
            LOSS: loss,
            LOSS_VALID: loss_valid,
            ACCURACY: accuracy * 100,
            ACCURACY_VALID: accuracy_valid * 100,
            BATCH_SIZE: self.trainer.batch_size  #
        }
        logging.debug("Sequence log: {}".format(data))
        self.sequence_data.add_data(data)

    def log_memory(self, status: Optional[Dict[str, str]]=None):
        memory_used, memory_allocated = self.memory()
        data = {
            EPOCH: self._epoch(status),
            CUDA_MEM_AVAIL: memory_allocated,
            CUDA_MEM_USE: memory_used
        }
        logging.debug("Memory log: {}".format(data))
        self.memory_data.add_data(data)

    def log_epoch(self) -> float:
        # Fetch losses and clear buffers
        loss, loss_valid = self.manager_epoch.loss(), self.manager_validation.loss()
        accuracy, accuracy_valid = self.manager_epoch.accuracy(), self.manager_validation.accuracy()
        self.manager_epoch.clear()
        self.manager_validation.clear()

        data = {
            EPOCH: int(self._epoch()),
            TIME: self.manager_epoch.time_elapsed.total_seconds(),
            LOSS: loss,
            LOSS_VALID: loss_valid,
            ACCURACY: accuracy * 100,
            ACCURACY_VALID: accuracy_valid * 100,
            BATCH_SIZE: self.trainer.batch_size  #
        }
        logging.warning("End of epoch log: {}".format(data))
        self.epoch_data.add_data(data)

        return accuracy_valid * 100

    def log_parameters(self):
        logging.info('Model total parameters: {}'.format(self.parameters()))

    def log_saved(self, file: BinaryIO):
        logging.warning('Model saved at {}'.format(file.name))

    def log_loaded(self, file: str):
        logging.warning('Model loaded from {}'.format(file))

    def parameters(self) -> int:
        # Calculating total number of parameters
        params = list(self.trainer.model.parameters()) + list(self.trainer.loss.parameters())
        total_params = sum(
            x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())

        return total_params

    def _epoch(self, status: Optional[Dict[str, str]]=None) -> float:
        if status is None:
            epoch_portion = 0

        else:
            example, examples = status['example'].split('/')
            epoch_portion = float(example) / float(examples)

        return self.trainer.epoch + epoch_portion

    @staticmethod
    def memory() -> Tuple[int, int]:
        if cuda.is_available():
            # Empty cache
            cuda.empty_cache()

            memory_used = cuda.memory_allocated()
            memory_allocated = cuda.max_memory_allocated()

            text_used = as_giga_mega_kilo_bytes(memory_used)
            text_allocated = as_giga_mega_kilo_bytes(memory_allocated)
            logging.debug("Freeing memory: {}/{} (used/allocated)".format(text_used, text_allocated))

            return memory_used, memory_allocated
