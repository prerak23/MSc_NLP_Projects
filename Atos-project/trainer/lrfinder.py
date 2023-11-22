import copy
import logging
import os
from typing import List, Optional, Tuple

from trainer import Trainer


class LearningRateFinder(object):
    """A tool to find the optimal learning rate for a model.

    Attr:
        trainer: trainer containing the model and all other training parameters; never modified.
        trainer_args and tracker_args: a copy of the parameters of the trainer
        backup_file: the path to the file where the backup of the model is created, then reloaded after each learning
        rate

    Usage:
        Basically you need to create/load a trainer, and create a LearningRateFinder with it and the necessary
        parameters to vary the learning rate.
        >>> trainer = Trainer(*args)
        >>> lr = LearningRateFinder(trainer, start_lr=1, stop_lr=1e-2, lr_multiplier=0.6).find_learning_rate()
    """

    def __init__(self, trainer: Trainer,
                 start_lr: Optional[float] = 1,
                 stop_lr: Optional[float] = 1e-5,
                 lr_multiplier: Optional[float] = 0.1):
        """

        :param trainer:
        :param start_lr:
        :param stop_lr:
        :param lr_multiplier:
        """
        self.trainer = trainer
        self.trainer_args = copy.copy(self.trainer.trainer_args)
        self.tracker_args = copy.copy(self.trainer.tracker_args)

        self.backup_file = "bkp.pth"

        # lr distribution
        self.start_lr = start_lr
        self.stop_lr = stop_lr

        assert self.start_lr > self.stop_lr
        self.lr_multiplier = lr_multiplier  # 0.1 pour start_lr * 10^-i

        assert 0 < self.lr_multiplier < 1

    def find_learning_rate(self, plot: Optional[bool]=True) -> Tuple[float, Trainer]:
        # save trainer backup
        self.save_backup()

        # train a model for each learning rate generated
        accuracies = []
        learning_rates = []
        for learning_rate in self.generate_learning_rates():
            logging.debug("Learning rate {} start".format(learning_rate))

            trainer = self.train(learning_rate)

            # evaluate model performance on validation corpus
            accuracy = self.evaluate(trainer)

            # save performance
            accuracies.append(accuracy)
            learning_rates.append(learning_rate)

            logging.debug("Learning rate {} result accuracy {}%".format(learning_rate, accuracy))

        # show results and choose learning rate
        learning_rate = self.log_plot_choose(learning_rates, accuracies, plot)

        # rebuild best trainer
        best_trainer = self.load_backup(learning_rate)

        # remove trainer backup
        self.remove_backup()

        return learning_rate, best_trainer

    def save_backup(self):
        with open(self.backup_file, mode="wb") as f:
            self.trainer.save(f)

    def load_backup(self, learning_rate: float) -> Trainer:
        self.trainer_args.learning_rate = learning_rate
        self.trainer_args.load_file = open(self.backup_file, mode="rb")

        # create trainer
        trainer = Trainer(self.trainer_args, self.tracker_args)

        return trainer

    def remove_backup(self):
        os.remove(self.backup_file)

    def train(self, learning_rate: float) -> Trainer:
        # load a copy of the trainer with a specific learning rate
        trainer = self.load_backup(learning_rate)

        # # run the trainer for a number of sequences
        # with trainer.tracker.epoch() as tracker:
        #     trainer.sequence = 0
        #     while trainer.sequence < trainer.sequences:  # TODO define epoch
        #         tracker.add(*trainer.single_batch())
        #
        #         trainer.sequence += trainer.batch_size
        #         trainer.batch += 1

        # run the trainer for a full epoch
        trainer.single_epoch()

        return trainer

    def evaluate(self, trainer: Trainer) -> float:
        trainer.validation()
        return trainer.tracker.log_epoch()

    def generate_learning_rates(self):
        lr = self.start_lr
        while lr > self.stop_lr:
            yield lr
            lr *= self.lr_multiplier

    def log_plot_choose(self, learning_rates: List[float], accuracies: List[float], plot: bool):
        from numpy import argmax

        min_id = int(argmax(accuracies))
        logging.warning("Best learning rate: {}, with accuracy {}".format(learning_rates[min_id], accuracies[min_id]))

        # plot

        import matplotlib.pyplot as plt

        plt.title('Learning rate with respect to accuracy.')
        plt.semilogx(learning_rates, accuracies)
        plt.axvline(learning_rates[min_id])
        plt.grid(True)

        if plot:
            plt.show()
        else:
            plt.savefig(os.path.join(self.trainer.tracker_args.save_folder, "lr_plot.png"))

        return learning_rates[min_id]
