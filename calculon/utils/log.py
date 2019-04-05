import sys
import os
import logging

LOG_COLORS = {
    'BLUE': '\033[94m',
    'DEFAULT': '\033[99m',
    'GREY': '\033[90m',
    'YELLOW': '\033[93m',
    'BLACK': '\033[90m',
    'CYAN': '\033[96m',
    'GREEN': '\033[92m',
    'DGREEN': '\033[32m',
    'MAGENTA': '\033[95m',
    'WHITE': '\033[97m',
    'RED': '\033[91m',
    'END': '\033[0m'
}

screen_logger = logging.getLogger("screen")
screen_logger.handlers = []
screen_logger.addHandler(logging.StreamHandler(sys.stdout))
screen_logger.setLevel(logging.INFO)
file_logger = logging.getLogger("file")
file_logger.handlers = []
file_logger.setLevel(logging.INFO)
log_color = True
log_timestamp = True


def setup(options):
    ''' all-in-one configure the logging module through a dict

    Args:
        options (dict): logger settings
    '''
    set_log_level(options['log_level'])
    set_colored_output(options['color'])
    set_screen_output(options['screen_output'])
    set_timestamp_output(options['log_timestamp'])
    set_file_output(options['log_file_path'])


def set_log_level(lvl):
    ''' Set the current log level to either 'error', 'warn', 'info' or 'debug'

    Args:
        lvl (str): the desired log level
    '''
    global screen_logger, file_logger
    numeric_level = getattr(logging, lvl.upper())
    screen_logger.setLevel(numeric_level)
    file_logger.setLevel(numeric_level)


def set_colored_output(is_colored):
    ''' Enable or disable colors for on screen output

    Args:
        is_colored (bool): set to True for colors
    '''
    global log_color
    log_color = True if is_colored else False


def set_screen_output(print_screen):
    ''' Enable or disable output to the screen

    Args:
        print_screen (bool): set to True for screen output
    '''
    global screen_logger
    if print_screen and not screen_logger.hasHandlers():
        screen_logger.addHandler(logging.StreamHandler(sys.stdout))
    elif not print_screen and screen_logger.hasHandlers():
        screen_logger.handlers = []


def set_timestamp_output(with_timestamp):
    ''' Enable or disable output of the time stamp for file output

    Args:
        with_timestamp (bool): set to True for time stamps in the file
    '''
    global log_timestamp
    log_timestamp = True if with_timestamp else False


def set_file_output(file_path):
    ''' Disable or enable and specify the path to a log file

    Args:
        file_path (str): set to None or a valid path
    '''
    global file_logger
    if file_path is not None and not file_logger.hasHandlers():
        if not os.path.isdir(os.path.dirname(file_path)):
            raise NotADirectoryError(
                "The target directory for the log file is not reachable")
        file_handler = logging.FileHandler(file_path)
        if log_timestamp:
            file_handler.setFormatter(logging.Formatter(
                fmt="%(asctime)s: %(message)s"))
        file_logger.addHandler(file_handler)
    elif file_path is None and file_logger.hasHandlers():
        file_logger.handlers = []


def _split_color_info(args):
    ''' Split and recombine possible color info into message '''
    if len(args) >= 2:
        sep = ''
        msg = []
        cmsg = []
        for item in args:
            try:
                color = LOG_COLORS[item]
                cmsg.append(color)
            except KeyError:
                msg.append(sep + item)
                cmsg.append(sep + item)
                sep = ' '
        if len(msg) < len(cmsg):
            cmsg.append(LOG_COLORS['END'])
        return (''.join(msg), ''.join(cmsg))
    msg = ' '.join(args[0:])
    return (msg, msg)


def _do_log(lvl, args):
    ''' Unified call of the underlying loggers.

    Args:
        lvl (logging.level): The log level
        args         (list): A list of strings

    Returns:
        str: The message w/out color codes
    '''
    plain_msg, color_msg = _split_color_info(tuple(map(str, args)))
    if log_color:
        screen_logger.log(lvl, color_msg)
    else:
        screen_logger.log(lvl, plain_msg)
    file_logger.log(lvl, plain_msg)
    return plain_msg


def error(*args):
    ''' Write log at ERROR level. The message can be prended by a color
    idenifier

    Args:
        *args (iterable): The first entry can optionally be a color identifier.
                          Everything else is treated as the message.
    '''
    return _do_log(logging.ERROR, args)


def warn(*args):
    ''' Write log at WARNING level. The message can be prended by a color
    idenifier

    Args:
        *args (iterable): The first entry can optionally be a color identifier.
                          Everything else is treated as the message.
    '''
    return _do_log(logging.WARN, args)


def info(*args):
    ''' Write log at INFO level. The message can be prended by a color
    idenifier

    Args:
        *args (iterable): The first entry can optionally be a color identifier.
                          Everything else is treated as the message.
    '''
    return _do_log(logging.INFO, args)


def debug(*args):
    ''' Write log at DEBUG level. The message can be prended by a color
    idenifier

    Args:
        *args (iterable): The first entry can optionally be a color identifier.
                          Everything else is treated as the message.
    '''
    return _do_log(logging.DEBUG, args)
