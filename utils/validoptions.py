from calculon.utils import log


class FailedCastError(ValueError):
    pass


class FailedVerificationError(RuntimeError):
    pass


class ValidOption(object):
    ''' Handles casting, verification and documentation of options.

    This class can be used in (nested) dicts in place of the value. It
    can then be used to cast or verify the value that gets written to
    its place. Furthermore this class allows the specification of
    documentation and helpers useful for creating a argparse parser.
    '''
    _valid_attrs = ['type', 'default', 'choices', 'help', 'subtype',
                    'length', 'alias']
    _required_attrs = ['type']

    def __init__(self, **kwargs):
        ''' Setup the valid options through keywords.

        The ValidOption objects are configured through the use of
        keywords. A required keyword is 'type', which specifies the
        option type and takes care of casting.
        Additional keywords may be:
        'choices': takes a list with valid options
        'help': specifies a help string,
        'subtype': gives the type of options in within a type=list
                   object
        'length': specifies the number of entries within a type=list
                  object
        'alias': specifies aliases for the option

        Example:
        An option 'age' that shall use integer values, have a default
        of 18 and be addressable under the alisa 'a' on the cmd line:
        {'age': ValidOption(type=int, default=18,
                            help='The persons age', alias=['--a'])}

        A option with a list of 2 specific strings:
        {'code_type': ValidOption(
                type=list,
                subtype=str,
                default=['python', 'messy'],
                length=2,
                choices=['python', 'c', 'c++', 'messy', 'clean'],
                help='Specifiy what best describes the code')}

        Args:
            kwargs: The keywords for the option configuration.

        Raises:
            RuntimeError: On missing or invalid options.
        '''
        # defaults
        self.default = None
        self.help = 'Not specified'
        self.alias = []
        # process kwargs
        for key in self._required_attrs:
            if key not in kwargs.keys():
                message = log.error('Missing required option {} to \
                                    ValidOptions'.format(key))
                raise KeyError(message)
        for key, arg in kwargs.items():
            if key not in self._valid_attrs:
                message = log.error('Specified unknown option {} to \
                                    ValidOptions'.format(key))
                raise KeyError(message)
            setattr(self, key, arg)
        # special cases
        if self.type == list and not hasattr(self, 'subtype'):
            message = log.error('ValidOptions with type list \
                                    require a subtype')
            raise RuntimeError(message)
        # verify the default type
        if self.default is not None:
            self.verify(self.default)

    def cast(self, val, ltype=None):
        ''' Cast to type, to subtype if type is a list.

        Args:
            val: The value to be cast.
            ltype: Manually specified type to cast to.

        Returns:
            The value, cast to the proper type.

        Raises:
            FailedCastError: If the value could not be cast.
        '''
        if ltype is None:
            ltype = self.type
        # special cases
        if ltype == bool:
            return self._cast_bool(val)
        elif ltype == list:
            return self._cast_list(val)
        # regular case
        try:
            return ltype(val)
        except ValueError:
            message = log.error('Unable to cast \'{}\' to \
                                type \'{}\''.format(val, ltype))
            raise FailedCastError(message)

    @staticmethod
    def _cast_bool(val):
        ''' Handle the bool separately, as too much can be true or false '''
        if (val is True or
                (isinstance(val, str) and val.lower() == 'true') or
                (isinstance(val, int) and val != 0)):
            return True
        elif (val is False or
              (isinstance(val, str) and val.lower() == 'false') or
              (isinstance(val, int) and val == 0)):
            return False
        message = log.error('Unable to cast \
                        \'{}\' to type \'bool\''.format(val))
        raise FailedCastError(message)

    def _cast_list(self, val):
        ''' Handle the casting of lists recursively '''
        if isinstance(val, list):
            return list(map(lambda x: self.cast(x, self.subtype), val))
        elif isinstance(val, str):
            mod = val.strip()
            if len(mod) >= 2 and mod[0] == '[' and mod[-1] == ']':
                mod = mod[1:-1].split(',')
                return list(map(lambda x: self.cast(x.strip(), self.subtype),
                                mod))
        message = log.error('Unable to cast \'{}\' \
                    to \type \'list({})\''.format(
                    val, self.subtype))
        raise FailedCastError(message)

    def verify(self, val):
        ''' Verify against choices, if available

        Args:
            val: The value to be verified

        Raises:
            FailedVerificationError: If the checks failed
        '''
        if self.type == list and not isinstance(val, list):
            message = log.error('\'{}\' is not a list'.format(val))
            raise FailedVerificationError(message)
        if hasattr(self, 'choices'):
            if self.type == list:
                if hasattr(self, 'length') and len(val) != self.length:
                    message = log.error('Invalid length of list \'{}\', \
                            should be {}'.format(val, self.length))
                    raise FailedVerificationError(message)
                for v in val:
                    if v not in self.choices:
                        message = log.error('Invalid selection \'{}\'. \
                                Possible choices are \'{}\''.format(
                                    val, self.choices))
                        raise FailedVerificationError(message)
            elif val not in self.choices:
                message = log.error('Invalid selection \'{}\'. \
                    Possible choices are \'{}\''.format(val, self.choices))
                raise FailedVerificationError(message)

    def nargs_argument(self):
        ''' Returns argument that can be used as input to argparse nargs '''
        if self.type == list:
            if hasattr(self, 'length'):
                return int(self.length)
            return '+'
        return None
