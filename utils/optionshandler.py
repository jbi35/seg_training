import sys
try:
    import simplejson as json
except ImportError:
    import json
import calculon.utils.log as log
from calculon.utils.validoptions import ValidOption
from calculon.utils.optionstree import OptionsTree
from calculon.utils.parserhandler import ParserHandler, PrintOptionsAction


class OptionsHandler(object):
    ''' This class handles all configuration options through defaults, files
    and cmd line arguments.

    Obtaining the actual currently selected configuration is a multi step
    process.
    In the first step all major commands with their respective default option
    trees are added to assemble the full options tree. This tree can be used
    to return defaults and cast and verify other options trees. Use the
    add_sub_command to add the options trees.
    The second stage is processing the different configurations sources. For
    this the cmd line handler is set up first. This is done by passing the
    valid options tree to the handler class which can then parse and verify
    cmd line arguments.
    There are now three sources of configuration trees. The default options
    provided by the options handler, an input file in JSON format that can be
    converted to the proper python types by the options handler and the
    options parsed from the command line.
    For the final configuration the defaults are overwritten by the config
    read from the file, which in turn is overwritte by the cmd line config.
    '''

    default_options = {
            'mode': ValidOption(
                type=str, default='single',
                choices=['single', 'batch'],
                help='Process one or multiple data sets'),
            'log_level': ValidOption(
                alias=['--lvl'], type=str, default='info',
                choices=['debug', 'info', 'warn', 'error'],
                help='More output'),
            'color': ValidOption(
                type=bool, default=True,
                help='Colored display output'),
            'screen_output': ValidOption(
                type=bool, default=True,
                help='Enable display output'),
            'log_timestamp': ValidOption(
                type=bool, default=True,
                help='Adds time stamps to the log output'),
            'log_file_path': ValidOption(
                type=str, default=None,
                help='The path to the log file'),
            'output_dir': ValidOption(
                type=str, default='./',
                help='Specifies the output directory'),
            'no_timestamp': ValidOption(
                type=bool, default=False,
                help='Omit the time stamp when creating output directories')}

    def __init__(self):
        ''' Prepares the overall defaults and argument parser. '''
        self.commands = []
        self.opt_tree = OptionsTree(self.default_options)

    def add_sub_command(self, cmd_name, defaults, cmd_aliases=None):
        ''' Add a sub command of the pipeline.

        The collected default options are updated and the parser is
        configured with the proper parameters.

        Args:
            cmd_name (string): The name under which a new group of settings
                can be addressed
            defaults (dict): A tree of dicts with ValidOption leaf-nodes
            cmd_aliases (list): A list of alias names for the cmd_name
        '''
        self.commands.append(cmd_name)
        self.opt_tree.add_sub_options(cmd_name, defaults)

    def get_defaults(self):
        ''' Return the default options.

        Returns:
            dict: a tree of options filled with defaults
        '''
        return self.opt_tree.get_defaults()

    def get_current_config(self, argv=sys.argv):
        ''' Returns the options after processing all configuration inputs.

        Args:
            argv (sliceable): a list of command line arguments

        Returns:
            dict: a tree of options in the final configuration
        '''
        argv.remove('--print-current-config')
        return self.process(argv)

    def process(self, argv=sys.argv):
        ''' Process the defaults, the config file and the cmd args
        into the final configuration.

        Args:
            argv (sliceable): a list of arguments to be parsed

        Return:
            dict: a hierarchy of dicts with the assembled configuration
        '''
        default_opts = self.opt_tree.get_defaults()

        flat_opt = self.opt_tree.get_flat_validoptions()
        parser_handler = ParserHandler(flat_opt)

        common = parser_handler.get_common_group()
        common.add_argument(
                '-i', '--input',
                help='Provide the configuration through a file')
        common.add_argument(
                '--print-defaults', action=PrintOptionsAction,
                options_callback=self.get_defaults,
                help='Print the defaults in JSON format')
        common.add_argument(
                '--print-current-config', action=PrintOptionsAction,
                options_callback=self.get_current_config,
                help='Print the current configuration in JSON format')
        common.add_argument(
                '--version', action='version',
                version='%(prog)s TBD')

        flat_args = parser_handler.parse_args(argv)
        full_args = self.opt_tree.inflate_flat_options(flat_args)
        full_args = self._handle_cmd_specification(full_args)

        input_opts = {}
        if 'input' in flat_args:
            input_opts = self.read_input(full_args['input'])
            full_args.pop('input')

        # check if input file and cmd line args are conflicting
        for current_key, current_val in input_opts.items():
            if current_val is None and current_key in full_args:
                msg = log.error('RED', 'The command line argument {} '
                                       'conflicts with the deactivation in '
                                       'the input file.'.format(current_key))
                raise ValueError(msg)

        self._recursive_update(default_opts, input_opts)
        self._recursive_update(default_opts, full_args)
        return default_opts

    def read_input(self, file_path):
        ''' Update the defaults with a loaded config.

        Args:
            file_path (string): path to a JSON file to read a config from

        Returns:
            dict: a hierarchy of dicts with the selected options
        '''
        in_config = {}
        with open(file_path) as ff:
            in_config = json.load(ff)
        return self.opt_tree.cast_and_verify(in_config)

    @classmethod
    def _recursive_update(cls, old, new):
        ''' Recursively update the old configuration hierarchy w/ a new one.

        Args:
            old (dict): a hierarchy of dicts to be updated
            new (dict): a hierarchy of dicts with which the prev config is
                updated
        '''
        for k, v in new.items():
            if k not in old:
                msg = log.error('RED', 'Tried to set the unknown option: {}\n'
                                       'Valid choices are {}'.format(k, old))
                raise KeyError(msg)
            if isinstance(new[k], dict) and isinstance(old[k], dict):
                cls._recursive_update(old[k], new[k])
            else:
                old[k] = v

    def _handle_cmd_specification(self, args):
        ''' Check and cleanup cmd line arguments.

        This methods checks whether one of major subcommands was used. If
        this is _not_ the case nothings is altered and the configurations
        is passed on as is. Should one or multiple major subcommands be
        specified, then all unused subcommands are actively disabled. This
        is achieved by setting them to None.

        Args:
            args (iterable): list of commandline arguments

        Returns:
            dict: dictionary of usable cmd line options
        '''
        cmd_used = set()
        for k in self.commands:
            if k in args.keys():
                cmd_used.add(k)
        if len(cmd_used) == 0:
            # no cmd specified, return as is
            return args
        # kill all cmds _not_specified
        cmd_unused = set(self.commands).difference(cmd_used)
        for k in cmd_unused:
            args[k] = None
        return args
