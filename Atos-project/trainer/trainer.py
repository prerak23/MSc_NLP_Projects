from model import Model
from trainer.loss import accuracy
from . import Loss, LossArgs, Optim, OptimArgs
from .tracker import Tracker, TrackerArgs

from data import get_corpus, Corpus

from typing import BinaryIO, Optional, Tuple, Union, List

from torch import save, load, no_grad, Tensor, equal, LongTensor

class TrainerArgs:
    def __init__(self,
                 corpus_name: str,
                 emb_out: int,
                 linear_layers: Union[List[Union[int, None]], None],
                 learning_rate: float,
                 weight_decay: float,
                 tracker_folder: str,
                 cuda: bool,
                 epochs: int,
                 checkpoint_file: Union[str, bytes],
                 memory_interval: int,
                 validation_interval: int,
                 load_file: Optional[BinaryIO] = None):
        self.corpus_name = corpus_name
        self.emb_out = emb_out
        self.linear_layers = linear_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.checkpoint_file = checkpoint_file
        self.load_file = load_file
        self.tracker_folder = tracker_folder
        self.cuda = cuda
        self.epochs = epochs

        self.validation_interval = validation_interval
        self.memory_interval = memory_interval


class Trainer(object):
    def __init__(self, trainer_args: TrainerArgs, tracker_args: TrackerArgs):
        # Always processed data
        self.trainer_args = trainer_args
        self.tracker_args = tracker_args
        self.last_validation = 0
        self.last_memory = 0
        self.batch = 0

        # Initialising tracker
        self.tracker = Tracker(self, self.tracker_args)

        # Data processed only once, reload on load
        if self.trainer_args.load_file is None:
            # Classical Trainer building
            self.corpus_name = self.trainer_args.corpus_name
            self.corpus = self.load_corpus()

            # Training loops values
            self.epoch = 0
            self.sequence = 0

            # Building model
            self.model = Model(len(self.corpus.dict.dictionary),
                               self.corpus.crop_pad.length,
                               self.trainer_args.emb_out,
                               self.trainer_args.linear_layers)

            # Checkpoint setting
            self.validation_accuracy_best = -1

            # First checkpoint
            with open(self.trainer_args.checkpoint_file, mode="wb") as f:
                self.save(f)

        else:
            # Load stored data
            load_data = load(self.trainer_args.load_file)

            # Recover corpus
            self.corpus_name = load_data["corpus"]
            self.corpus = self.load_corpus()

            # Training loops values
            self.epoch = load_data["epoch"]
            self.sequence = load_data["sequence"]
            self.validation_accuracy_best = load_data["valid accuracy (best)"]

            # Rebuild model
            self.model = Model(len(self.corpus.dict.dictionary),
                               self.corpus.crop_pad.length,
                               self.trainer_args.emb_out,
                               self.trainer_args.linear_layers)
            self.model.load_state_dict(load_data["model"])

            # Log load success
            self.tracker.log_loaded(self.trainer_args.load_file.name)

        # Batch size
        self.batch_size = self.corpus.batch.batch_size

        # Preparing loss and optimizer arguments
        optim_args = OptimArgs(self.model, self.trainer_args.learning_rate, self.trainer_args.weight_decay)
        loss_args = LossArgs(self.corpus.dict.dictionary)

        # Creating loss and optimiser
        self.optimizer = Optim(optim_args)
        self.loss = Loss(loss_args)

        if trainer_args.cuda:
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()

        # Log total parameters
        self.tracker.log_parameters()

    def load_corpus(self) -> Corpus:
        return get_corpus(self.corpus_name)

    def save_corpus(self) -> str:
        return self.corpus_name

    def save(self, file: BinaryIO):
        save({"model": self.model.state_dict(),
              "corpus": self.save_corpus(),
              "epoch": self.epoch,
              "sequence": self.sequence,
              "valid accuracy (best)": self.validation_accuracy_best}, file)
        self.tracker.log_saved(file)

    ####################################################################################################################

    def all_epochs(self):
        while self.epoch < self.trainer_args.epochs:
            self.single_epoch()

    def single_epoch(self):
        with self.tracker.epoch() as tracker:
            for data, target, status in self.corpus.train(True, self.trainer_args.cuda):
                #print(status)
                # Run training sequence
                tracker.add(*self.single_batch(data, target))

                # Validation
                if self.sequence == 0 or self.sequence - self.last_validation >= self.trainer_args.validation_interval:
                    self.last_validation = self.sequence
                    self.validation()
                    self.tracker.log_sequence(status)

                if self.sequence == 0 or self.sequence - self.last_memory >= self.trainer_args.memory_interval:
                    self.last_memory = self.sequence
                    self.validation()
                    self.tracker.log_memory(status)

                self.sequence += self.batch_size  # TODO
                self.batch += 1

        # Reset sequence count
        self.sequence = 0

        # Increment epoch count
        self.epoch += 1

        # Log end-of-epoch information
        self.validation()
        self.tracker.log_memory()
        validation_accuracy = self.tracker.log_epoch()

        # Save checkpoint if it performs better than previous best checkpoint
        if self.validation_accuracy_best < validation_accuracy:
            self.validation_accuracy_best = validation_accuracy
            with open(self.trainer_args.checkpoint_file, mode="wb") as f:
                self.save(f)

    def single_batch(self, data: LongTensor, target: LongTensor) -> Tuple[float, float]:
        with self.tracker.sequence() as tracker:
            # Adapt learning rate to sequence length
            learning_rate = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['lr'] = learning_rate * self.batch_size

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            result = self.model(data)
            loss = self.loss(result, target)

            # Store loss and accuracy
            loss_value = float(loss)
            accuracy_value = accuracy(result, target)
            tracker.add(loss_value, accuracy_value)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Resetting learning rate to initial value
            self.optimizer.param_groups[0]['lr'] = learning_rate

        return loss_value, accuracy_value

    def validation(self):
        with self.tracker.validation() as tracker:
            with no_grad():
                for data, target in self.corpus.valid(False, self.trainer_args.cuda):
                    # Forward pass
                    result = self.model(data)
                    loss = self.loss(result, target)

                    # Store loss
                    tracker.add(float(loss), accuracy(result, target))

    def test(self) -> Tuple[float, float, float]:
        with self.tracker.test() as tracker:
            with no_grad():
                for data, target in self.corpus.test(False, self.trainer_args.cuda):
                    # Forward pass
                    result = self.model(data)
                    loss = self.loss(result, target)

                    # Store loss
                    tracker.add(float(loss), accuracy(result, target))

        return self.tracker.log_test()
