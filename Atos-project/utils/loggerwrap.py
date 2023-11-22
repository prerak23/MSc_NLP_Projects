import sys
import logging.config
import logging.handlers

"""Logger wrapper"""
levels = {
    "disabled": None,
    "notset": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,

    "all": logging.NOTSET,
    "log interval": logging.INFO,
    "epoch": logging.WARNING,
}


def init_log(filename="logs/log.log", file_level="disabled", console_level="disabled"):
    # Getting date

    format_default = '%(asctime)s %(levelname)-8s %(message)s'
    format_debug = "%(asctime)s %(levelname)-8s %(filename)s:%(lineno)d: %(message)s"

    def formatter(level: str) -> str:
        return 'debug' if levels[level] == levels['debug'] else 'default'

    handlers = {}
    if console_level and levels[console_level]:
        handlers['console'] = {'class': 'logging.StreamHandler',
                               'formatter': formatter(console_level),
                               'level': levels[console_level]}

    if file_level and levels[file_level]:
        print("Logs are written at {}".format(filename))
        handlers['file'] = {'class': 'logging.FileHandler',
                            'filename': filename,
                            'mode': 'w',
                            'formatter': formatter(file_level),
                            'level': levels[file_level]}

    if console_level and levels[console_level] and file_level and levels[file_level]:
        global_level = min(levels[console_level], levels[file_level])
    elif file_level and levels[file_level]:
        global_level = levels[file_level]
    elif console_level and levels[console_level]:
        global_level = levels[console_level]
    else:
        global_level = levels["notset"]

    log_config = {'version': 1,
                  'formatters': {'default': {'format': format_default},
                                 'debug':   {'format': format_debug}},
                  'handlers': handlers,
                  'root': {'handlers': tuple(handlers.keys()), 'level': global_level}}
    logging.config.dictConfig(log_config)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception
