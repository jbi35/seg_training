import copy

from calculon.utils.validoptions import ValidOption
from calculon.utils import log


class InvalidKeyError(RuntimeError):
    pass


class OptionsTree(object):
    ''' This class handles the tree of valid configuration options.

    This class works by passing in trees with ValidOption objects as
    leaf-nodes. These trees are combined to the full tree of options.
    Use the __init__ and add_sub_options methods for assembling the tree.
    This tree can be used for several functions.

    Firstly, it can return the full default configuration by querying the
    ValidOption objects for the defaults. Use the get_defaults method to
    obtain it.

    Secondly a tree with a choice of options can be verified against the
    valid options tree. Additionally it handles handles casting to the
    correct python types. This is usefull for feeding in trees read from
    JSON files, which inherently store all values as strings. Use the
    cast_and_verify method for this functionality. If it does not raise
    an exception it will return a properly typed tree of options.

    The third functionality simply reformats the tree keys to strings
    that can be used as cmd line options. The first level is retained.
    For example the config
    {'first': {'second': {'third': 8}, 'secondandhalf': 59}}
    will be transformed to
    {'first': {'second:third': 8, 'secondandhalf': 59}}
    Use get_flat_validoptions for this transformation and
    inflate_flat_options to reverse this process.
    '''

    def __init__(self, base_options):
        ''' Set up the options tree with the base/root level options.

        Args:
            base_options (dict): a dict with validoptions at the base level
        '''
        self.collected_options = copy.deepcopy(base_options)

    def add_sub_options(self, category, options):
        ''' Add option trees for the major sub commands.

        Args:
            category (string): name of the sub command
            options (dict): hierarchy of dicts with validoptions
        '''
        self.collected_options[category] = copy.deepcopy(options)

    def get_defaults(self):
        ''' Traverses the tree and returns all defaults.

        Returns:
            dict: hierarchy of dicts terminating in the default values
        '''
        return self._recurse_validoption_attrs(self.collected_options,
                                               'default')

    @classmethod
    def _recurse_validoption_attrs(cls, opts, attr_name):
        ''' Recursive tree traversal to find the specified attribute.

        Args:
            opts (dict, ValidOption): subtree or valid option
            attr_name (string): name of the attribute to call

        Returns:
            dict: hierarchy of dicts terminating in the selected attribute
        '''
        if isinstance(opts, ValidOption):
            return getattr(opts, attr_name)
        ret = {}
        for k, v in opts.items():
            ret[k] = cls._recurse_validoption_attrs(v, attr_name)
        return ret

    def cast_and_verify(self, config):
        ''' Verify and if necessary cast the passed tree of options.

        Args:
            config (dict): hierarchy of dicts to be verified

        Returns:
            dict: hierarchy of dicts with cast inputs
        '''
        return self._recurse_cast_verify(self.collected_options, config, [])

    @classmethod
    def _recurse_cast_verify(cls, validator, config, trace):
        ''' Recurse the tree and validate/cast all leaves.

        Args:
            validator (dict): tree of dicts to verify against
            config (dict): tree of dicts to be verified/cast
            trace (list): the current stack of options for meaningfull error

        Returns:
            dict: tree with cast leaves
        '''
        if isinstance(validator, ValidOption):
            converted = validator.cast(config)
            validator.verify(converted)
            return converted
        ret = {}
        for k, v in config.items():
            try:
                ret[k] = cls._recurse_cast_verify(validator[k], v, [k]+trace)
            except KeyError as error:
                message = log.error('Invalid key\'{}\' in option\'{}\''.format(
                    k, ':'.join(reversed(trace))))
                raise InvalidKeyError(message) from error
        return ret

    def get_flat_validoptions(self):
        ''' Returns a mostly flat dict of ValidOptions w/ 1 level depth.

        All options past the first level are represented by a concatenation of
        the option strings separated by ':'. This is required for cmd line
        args as these have to be single flat strings.

        Returns:
            dict: The options with keys as flat strings
        '''
        full_dict = {}
        for key, val in self.collected_options.items():
            if isinstance(val, ValidOption):
                full_dict[key] = val
            else:
                full_dict[key] = self._recurse_flatten([], val)
        return full_dict

    @classmethod
    def _recurse_flatten(cls, current, val):
        ''' Recursively flatten a hierarchy of options.

        Args:
            current (list): current stack of options
            val (dict): subtree thats left

        Returns:
            dict: a single level dict with flattened options
        '''
        if isinstance(val, ValidOption):
            return {':'.join(current): val}
        ret = {}
        for k, v in val.items():
            ret.update(cls._recurse_flatten(current+[k], v))
        return ret

    @classmethod
    def inflate_flat_options(cls, raw_args, args=None):
        ''' Convert keys given with a ':' separator to nested dicts.

        Args:
            raw_args (dict): the dict with the flattened options
            args (dict): the dict to inflate into

        Returns:
            dict: the nested inflated dict with the options
        '''
        if args is None:
            args = {}
        for full_key, v in raw_args.items():
            k_split = full_key.split(':')
            k_last = k_split[-1]
            sub_args = cls._traverse_dict(k_split[:-1], args)
            if isinstance(v, dict):
                if k_last not in sub_args:
                    sub_args[k_last] = {}
                cls.inflate_flat_options(v, sub_args[k_last])
            else:
                sub_args[k_last] = v
        return args

    @classmethod
    def _traverse_dict(cls, keys, args):
        ''' Traverse nested dicts and create inner dicts if necessary.

        Args:
            keys (iterable): the keys to be added at this level
            args (dict): the dict the keys and nested dicts are added to

        Returns:
            dict: the nested dicts
        '''
        for k in keys:
            if k not in args:
                args[k] = {}
            args = args[k]
        return args
