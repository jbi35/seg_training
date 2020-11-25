import sys
import tensorflow as tf
from time import time, strftime, gmtime
import calculon.utils.log as log
import calculon.utils.optionshandler as opthand
from calculon.training.launcher import Launcher


def main(options):
    ''' Launches calculon'''
    # Get current time and logo
    t_main = time()
    the_logo()

    # Send options from json to Trainer
    Trainer_options = options
    my_Trainer = Launcher.from_options(Trainer_options)

    # If tfrecords path is not empty, start training immediately.
    # If tfrecords path is empty and csv path is not empty, create tfrecords.
    # If tfrecords path is empty and csv path is empty, first create csv from database,
    # then create tfrecords.
    my_Trainer.check_data_source()

    # Launch the training process
    #my_Trainer.launch()
    my_Trainer.launch_3d()
    # Print duration
    print("Duration(HH:MM:SS):", strftime("%H:%M:%S", gmtime(time() - t_main)))

def the_logo():
    ''' print the logo '''
    log.info('MAGENTA', '============================')
    log.info('MAGENTA', '===   BEHOLD CALCULON    ===')
    log.info('MAGENTA', '============================')


if __name__ == '__main__':

    opt_handler = opthand.OptionsHandler()
    opt_handler.add_sub_command('ct_database',
                                Launcher.ct_database_options)
    opt_handler.add_sub_command('tfrecords',
                                Launcher.tfrecords_options)
    opt_handler.add_sub_command('training',
                                Launcher.training_options)
    options = opt_handler.process()
    log.setup(options)
    sys.exit(main(options))
