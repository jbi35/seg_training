import sys
import argparse
try:
    import simplejson as json
except ImportError:
    import json

from calculon.utils.validoptions import ValidOption
from calculon.utils import log


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

    message = log.error('Boolean value expected.')
    raise argparse.ArgumentTypeError(message)


class PrintOptionsAction(argparse.Action):
    def __init__(self,
                 option_strings,
                 options_callback=None,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 help='show options as JSON'):
        super(PrintOptionsAction, self).__init__(
                option_strings=option_strings,
                dest=dest,
                default=default,
                nargs=0,
                help=help)
        self.options_callback = options_callback

    def __call__(self, parser, namespace, values, option_string=None):
        options_callback = self.options_callback
        if options_callback is None:
            options_callback = parser.options_callback
        options = options_callback()
        options_text = json.dumps(options, indent=4)
        parser._print_message(options_text, sys.stdout)
        parser.exit()


class ParserHandler(object):

    def __init__(self, validoption_config):
        ''' Sets up argparse from the flattish dict of ValidOptions passed '''
        self.base_parse = argparse.ArgumentParser(
                description='Generate a lung model')
        self.common = self.base_parse.add_argument_group(
                'Common options',
                'These options apply to all pipeline steps')
        self.sub_parse = self.base_parse.add_subparsers(
                help='Select the pipeline steps to be executed')

        for key, val in validoption_config.items():
            if isinstance(val, ValidOption):
                self._add_argument(self.common, '--'+str(key), val)
            else:
                parser = self.sub_parse.add_parser(key)
                for skey, sval in val.items():
                    self._add_argument(parser, '--'+str(skey), sval)

    @staticmethod
    def _add_argument(parser, tag, validopt):
        argtype = validopt.type
        if argtype == list:
            parser.add_argument(tag, *validopt.alias, nargs='+', type=argtype,
                                help=validopt.help)
        elif argtype == bool:
            parser.add_argument(tag, *validopt.alias, type=str2bool,
                                help=validopt.help)
        else:
            parser.add_argument(tag, *validopt.alias, type=argtype,
                                help=validopt.help)

    def get_common_group(self):
        return self.common

    def parse_args(self, argv=sys.argv):
        ''' Parse the cmd line args and return a flattish dict with the
        corresponding options
        '''
        parsed_args = self._parse_multi_args(argv)
        return self._unpack_namespace(parsed_args)

    def _parse_multi_args(self, argv):
        ''' Trick argparse into handling multiple sub-commands '''
        # Divide argv by commands
        split_argv = [[]]
        for c in argv[1:]:
            if c in self.sub_parse.choices:
                split_argv.append([c])
            else:
                split_argv[-1].append(c)
        # Initialize namespace
        args = argparse.Namespace()
        for c in self.sub_parse.choices:
            setattr(args, c, None)
        # Parse each command
        self.base_parse.parse_args(split_argv[0], namespace=args)
        for argv in split_argv[1:]:
            n = argparse.Namespace()
            setattr(args, argv[0], n)
            self.base_parse.parse_args(argv, namespace=n)
        return args

    @staticmethod
    def _unpack_namespace(args):
        ''' Convert namespace to dict and strip Nones '''
        outer = dict([(k, v) for k, v in vars(args).items() if v is not None])
        for k, v in outer.items():
            if isinstance(v, argparse.Namespace):
                outer[k] = dict(
                        [(k, v) for k, v in vars(v).items() if v is not None])
        return outer
