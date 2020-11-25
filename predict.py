# standard imports
import sys
from time import time
import calculon.utils.log as log
import calculon.utils.optionshandler as opthand
from calculon.inference.inference import \
    Inference


def main(options):
    ''' do cool stuff'''

    t_main = time()
    the_logo()

    # Send options from json to Tester
    Inference_options = options
    my_Inference = Inference.from_options(Inference_options)

    # Execute
    my_Inference.execute()


def the_logo():
    ''' print the logo '''
    log.info('MAGENTA', '============================')
    log.info('MAGENTA', '===   BEHOLD CALCULON    ===')
    log.info('MAGENTA', '============================')


if __name__ == '__main__':

    opt_handler = opthand.OptionsHandler()
    opt_handler.add_sub_command('inference',
                                Inference.default_options,
                                ['s'])
    options = opt_handler.process()
    log.setup(options)
    sys.exit(main(options))
