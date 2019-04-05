# standard imports
import sys
from time import time

import calculon.utils.log as log
import calculon.utils.optionshandler as opthand
from calculon.pre_processing.pre_processor import \
    Preprocessor


def main(options):
    ''' do cool stuff'''

    t_main = time()
    the_logo()

    pre_processing_options = options['pre_processing']
    my_preprocessor = Preprocessor.from_options(pre_processing_options)

    my_preprocessor.augment_data("dummy_path", "dummy_path_2")
    mode = options.get('mode', 'batch')


def the_logo():
    ''' print the logo '''
    log.info('MAGENTA', '============================')
    log.info('MAGENTA', '===   BEHOLD CALCULON    ===')
    log.info('MAGENTA', '============================')


if __name__ == '__main__':
    opt_handler = opthand.OptionsHandler()
    opt_handler.add_sub_command('pre_processing',
                                Preprocessor.default_options,
                                ['s'])
    options = opt_handler.process()
    log.setup(options)
    sys.exit(main(options))
