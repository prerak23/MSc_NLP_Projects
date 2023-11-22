import logging
from os.path import join, exists
from os import makedirs
from typing import Tuple, Union, Iterator, List

import utils.loggerwrap as log

from argparse import ArgumentParser, Namespace

from trainer import TrainerArgs, Trainer
from trainer.lrfinder import LearningRateFinder
from trainer.tracker import TrackerArgs

FilePath = Union[str, bytes]

########################################################################################################################
#   Arguments default values                                                                                           #
########################################################################################################################
_MODEL_EMB_SIZE = 126
_MODEL_LAYERS = []

_CORPUS_NAME = "firsttest"

_FILES_ROOT = "save"
_FILES_SUB_FOLDER = ""
_FILES_CHECKPOINT = "checkpoint.pth"
_FILES_LOAD = None
_FILES_ARGS = "args.pkl"

_TRAINING_MINI_BATCH_SIZE = 128
_TRAINING_EPOCHS = 10
_TRAINING_LEARNING_RATE = 0.2
_TRAINING_WEIGHT_DECAY = 1.2e-6
_TRAINING_CUDA = True

_TRACKER_BUFFER_SIZE = 10
_TRACKER_MEMORY_INTERVAL = 500
_TRACKER_VALID_INTERVAL = 1000

_LOG_FILE = "log.txt"
_LOG_LEVELS = log.levels.keys()
_LOG_FILE_LEVEL = "disabled"
_LOG_CONSOLE_LEVEL = "disabled"

_LEARNING_RATE_MIN = 1e-5
_LEARNING_RATE_MAX = 1
_LEARNING_RATE_MULTIPLIER = 0.2


########################################################################################################################
#   Argument file I/O                                                                                                  #
########################################################################################################################
def save_args(args: Namespace, save_folder: FilePath) -> None:
    with open(join(save_folder, _FILES_ARGS), 'wb') as f:
        import pickle
        pickle.dump(args, f)
        logging.info("Args saved at {}".format(f.name))


def load_args(save_folder: FilePath) -> Namespace:
    with open(join(save_folder, _FILES_ARGS), 'rb') as f:
        import pickle
        return pickle.load(f)


########################################################################################################################

def hidden_layer_list(argument: str) -> List[int]:
    """Convert a list of numbers separated by ',' to a list of values"""
    return list(map(int, argument.split(',')))  # 3.x: consider wrapping in list()


def parse_args() -> Namespace:
    """Create the necessary parser and parse arguments

    :return: The object returned by parser.parse_args()
    """
    parser = ArgumentParser(description='PyTorch NN Language Model')

    # Corpus parameter
    parser.add_argument('-C', '--corpus',
                        default=_CORPUS_NAME, type=str,
                        help='name of the corpus to load')

    # Corpus parameter
    parser.add_argument('--no-cuda',
                        default=True, action='store_false', dest='cuda',
                        help='disables cuda')

    # Logger arguments group
    logger_parser = parser.add_argument_group()
    logger_parser.add_argument('-c', '--log-console-level',
                               default=_LOG_CONSOLE_LEVEL, choices=_LOG_LEVELS, type=str,
                               help='logging level of the log console')
    logger_parser.add_argument('-f', '--log-file-level',
                               default=_LOG_FILE_LEVEL, choices=_LOG_LEVELS, type=str,
                               help='logging level of the log file')
    logger_parser.add_argument('-F', '--log-file',
                               default=_LOG_FILE, type=str,
                               help='file name of the log file')
    logger_parser.add_argument('-t', '--tracker-buffer-size',
                               default=_TRACKER_BUFFER_SIZE, type=int,
                               help='size of the data_bkp writing buffer of the data_bkp tracker')
    logger_parser.add_argument('-m', '--memory-interval',
                               default=_TRACKER_MEMORY_INTERVAL, type=int,
                               help='number of sequences/batches between two memory logs')
    logger_parser.add_argument('-v', '--validation-interval',
                               default=_TRACKER_VALID_INTERVAL, type=int,
                               help='number of sequences/batches between two validations')

    # Training arguments group
    training_parser = parser.add_argument_group()
    # training_parser.add_argument('-b', '--mini-batch-size',
    #                              default=_TRAINING_MINI_BATCH_SIZE, type=int,
    #                              help='batch size')
    training_parser.add_argument('-e', '--epochs',
                                 default=_TRAINING_EPOCHS, type=int,
                                 help='upper epoch limit')
    training_parser.add_argument('-r', '--learning-rate',
                                 default=_TRAINING_LEARNING_RATE, type=float,
                                 help='learning rate')
    training_parser.add_argument('-w', '--weight-decay',
                                 default=_TRAINING_WEIGHT_DECAY, type=float,
                                 help='weight decay')  # TODO explain a bit

    # Model arguments group
    model_parser = parser.add_argument_group()
    model_parser.add_argument('-E', '--embedding-size',
                              default=_MODEL_EMB_SIZE, type=int,
                              help='size of the output of the embeddings layer')
    model_parser.add_argument('-H', '--hidden-layers',
                              default=_MODEL_LAYERS, type=hidden_layer_list,
                              help='size and number of hidden layers, use format "50,30,40" to have three hidden layers'
                                   ' of respective size 50, 30 and 40, starting from the input side; use any number'
                                   ' under 1 as a wild card to use the size of the previous layer, like in "-1,-1,42"')

    # Save sub-folder and files arguments group
    file_parser = parser.add_argument_group()
    file_parser.add_argument('-s', '--files-sub-folder',
                             default=_FILES_SUB_FOLDER, type=str,
                             help='sub-folder of default save folder in which save and load model, arguments, '
                                  'and statistics')
    file_parser.add_argument('--checkpoint', type=str, default=_FILES_CHECKPOINT,
                             help='checkpoint file name')

    # Load arguments group
    load_parser = file_parser.add_mutually_exclusive_group()
    load_parser.add_argument('--load-checkpoint',
                             default=False, action='store_true',
                             help='load last checkpoint')
    load_parser.add_argument('--load-file',
                             default=_FILES_LOAD, type=str,
                             help='load specific file name')

    # Training/learning rate fine-tuning arguments group
    finetune_lr_parser = file_parser.add_argument_group()
    finetune_lr_parser.add_argument('-T', '--no-training',
                                    default=True, action='store_false', dest='train',
                                    help='disable training')
    finetune_lr_parser.add_argument('-R', '--fine-tune-learning-rate',
                                    default=False, action='store_true',
                                    help='fine-tune learning rate; when \'-T\' is used, show a plot of the learning'
                                         'rate, then exits; otherwise, find the optimal rate then train the model with '
                                         'it')
    finetune_lr_parser.add_argument('--learning-rate-max',
                                    default=_LEARNING_RATE_MAX, type=float,
                                    help='maximal learning rate when searching optimal learning rate')
    finetune_lr_parser.add_argument('--learning-rate-min',
                                    default=_LEARNING_RATE_MIN, type=float,
                                    help='minimal learning rate when searching optimal learning rate')
    finetune_lr_parser.add_argument('--learning-rate-multiplier',
                                    default=_LEARNING_RATE_MULTIPLIER, type=float,
                                    help='multiplier of the learning rate when searching optimal learning rate')

    # Parse parameters
    args = parser.parse_args()
    args.tied = True

    return args


def process_args(args: Namespace) -> Tuple[Namespace, FilePath, Union[FilePath, None]]:
    """Process data_bkp in args, and adjust dependant arguments

    :param args: Namespace object containing the arguments to process
    :return: Namespace object containing the processed arguments, the root folder path, and an eventual path to the file
    from which load the model
    """
    save_folder = join(_FILES_ROOT, args.files_sub_folder)

    # Create necessary directories if they do not exist yet
    for directory in [""]:
        if not exists(join(save_folder, directory)):
            makedirs(join(save_folder, directory))

    # Adapt paths to the save_folder
    args.checkpoint = join(save_folder, args.checkpoint)
    args.log_file = join(save_folder, args.log_file)

    # Load checkpoint
    if not args.load_file and args.load_checkpoint:
        load_file = args.checkpoint

    elif args.load_file:
        assert exists(args.load_file) or exists(join(save_folder, args.load_file)), (
            "None of those files exist:\n{}\n{}".format(join(save_folder, args.load_file), args.load_file))

        if exists(join(save_folder, args.load_file)):
            args.load_file = join(save_folder, args.load_file)

        load_file = args.load_file

    else:
        load_file = None

    return args, save_folder, load_file


def start_logger(args: Namespace) -> None:
    """Setup the logger

    :param args: Namespace object containing the arguments of the logger
    """
    log.init_log(args.log_file,
                 file_level=args.log_file_level,
                 console_level=args.log_console_level)


def main(args: Namespace, save_folder: FilePath, load_file: Union[FilePath, None]) -> None:
    """Build necessary objects and call the script corresponding to the arguments

    :param args: Namespace object containing the arguments
    :param save_folder: path to the folder were every file will be saved
    :param load_file: optional path to the file to load
    """

    file = None
    try:
        if load_file:
            file = open(load_file, mode="rb")

        trainer_args = TrainerArgs(corpus_name=args.corpus,
                                   emb_out=args.embedding_size,
                                   linear_layers=args.hidden_layers,
                                   learning_rate=args.learning_rate,
                                   weight_decay=args.weight_decay,
                                   tracker_folder=save_folder,
                                   cuda=args.cuda,
                                   epochs=args.epochs,
                                   checkpoint_file=args.checkpoint,
                                   memory_interval=args.memory_interval,
                                   validation_interval=args.validation_interval,
                                   load_file=file)

        tracker_args = TrackerArgs(save_folder=save_folder, data_buffer_size=args.tracker_buffer_size)

        # Create trainer
        trainer = Trainer(trainer_args, tracker_args)

        if args.fine_tune_learning_rate:
            # Start learning rate fine-tuneing
            lr_finder = LearningRateFinder(trainer,
                                           start_lr=args.learning_rate_max,
                                           stop_lr=args.learning_rate_min,
                                           lr_multiplier=args.learning_rate_multiplier)

            # find learning rate and replace trainer with an updated version with optimal learning rate
            lr, trainer = lr_finder.find_learning_rate(plot=not args.train)

        if args.train:
            # Start training
            trainer.all_epochs()

    finally:
        if file is not None:
            file.close()


if __name__ == "__main__":
    # Parse args
    args = parse_args()

    # Process args
    args, save_folder, load_file = process_args(args)

    # Save args
    save_args(args, save_folder)

    # Start logger
    start_logger(args)

    # Run script
    main(args, save_folder, load_file)
