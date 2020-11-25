import pytest
import copy
import os
import tempfile
try:
    import simplejson as json
except ImportError:
    import json

from calculon.utils.optionshandler import OptionsHandler
from calculon.utils.validoptions import ValidOption


@pytest.fixture
def opt_handler_two_commands():
    cmd_opts1 = {'a': ValidOption(type=str, default='uiae'),
                 'b': ValidOption(type=int, default=42),
                 'c': {'d': ValidOption(type=int, default=1337)}}
    cmd_opts2 = {'one': ValidOption(type=str, default='uiae')}
    oh = OptionsHandler()
    oh.add_sub_command('test_cmd', cmd_opts1, ['t', 'tc'])
    oh.add_sub_command('other_cmd', cmd_opts2, ['oc'])
    return oh


@pytest.fixture
def sample_input_file():
    opts = {'test_cmd': {'b': '666'}, 'other_cmd': {'one': 'asdf'}}
    fd, file_path = tempfile.mkstemp(suffix='.json')
    with open(file_path, 'w') as ff:
        json.dump(opts, ff)
    yield file_path
    os.close(fd)
    os.remove(file_path)


@pytest.fixture
def sample_input_file_broken():
    opts = '{"test_cmd": {"b": "666"}, "other_cmd": }'
    fd, file_path = tempfile.mkstemp(suffix='.json')
    with open(file_path, 'w') as ff:
        ff.write(opts)
    yield file_path
    os.close(fd)
    os.remove(file_path)


@pytest.fixture
def basic_config_dict():
    return copy.deepcopy(
        {'a': 4, 'b': 13, 'c': {'c': 'c', 'd': -12.3, 'e': {'f': 42.0},
                                'g': 1337.0}, 'h': 'uiae'})


@pytest.fixture(scope='module')
def reference_default_config():
    return {'mode': 'single', 'log_level': 'info', 'output_dir': './',
            'color': True, 'log_file_path': None, 'log_timestamp': True,
            'screen_output': True, 'no_timestamp': False}


def test_get_defaults(reference_default_config):
    oh = OptionsHandler()
    result = oh.get_defaults()
    assert result == reference_default_config


def test_get_current_config(opt_handler_two_commands, sample_input_file,
                            reference_default_config):
    oh = opt_handler_two_commands
    with pytest.raises(ValueError):
        oh.get_current_config(['PROG.py', '--input', sample_input_file,
                               'test_cmd', '--b', '123'])
    result = oh.get_current_config(['PROG.py', '--print-current-config',
                                    '--input', sample_input_file,
                                    'test_cmd', '--b', '123'])
    assert result == {'other_cmd': None,
                      'test_cmd': {'a': 'uiae', 'b': 123, 'c': {'d': 1337}},
                      **reference_default_config}


def test_add_sub_command0(reference_default_config):
    cmd_opts = {'a': ValidOption(type=str, default='uiae'),
                'b': ValidOption(type=int, default=42),
                'c': {'d': ValidOption(type=int, default=1337)}}
    oh = OptionsHandler()
    oh.add_sub_command('test_cmd', cmd_opts)
    result = oh.get_defaults()
    assert result == {'test_cmd': {'a': 'uiae', 'b': 42, 'c': {'d': 1337}},
                      **reference_default_config}


def test_add_sub_command1(reference_default_config):
    cmd_opts = {'a': ValidOption(type=str, default='uiae'),
                'b': ValidOption(type=int, default=42),
                'c': {'d': ValidOption(type=int, default=1337)}}
    oh = OptionsHandler()
    oh.add_sub_command('test_cmd', cmd_opts, cmd_aliases=['tc'])
    result = oh.get_defaults()
    assert result == {'test_cmd': {'a': 'uiae', 'b': 42, 'c': {'d': 1337}},
                      **reference_default_config}


def test_read_input(opt_handler_two_commands, sample_input_file):
    oh = opt_handler_two_commands
    result = oh.read_input(sample_input_file)
    assert result == {'test_cmd': {'b': 666}, 'other_cmd': {'one': 'asdf'}}


def test_read_input_broken(opt_handler_two_commands, sample_input_file_broken):
    oh = opt_handler_two_commands
    with pytest.raises(json.decoder.JSONDecodeError):
        oh.read_input(sample_input_file_broken)


def test_process_args(opt_handler_two_commands, reference_default_config):
    oh = opt_handler_two_commands
    result = oh.process(['PROG.py', 'test_cmd', '--b', '123'])
    assert result == {'other_cmd': None,
                      'test_cmd': {'a': 'uiae', 'b': 123, 'c': {'d': 1337}},
                      **reference_default_config}


def test_process_args_input(opt_handler_two_commands, sample_input_file,
                            reference_default_config):
    oh = opt_handler_two_commands
    result = oh.process(['PROG.py', '--input', sample_input_file, 'test_cmd',
                         '--b', '123'])
    assert result == {'other_cmd': None,
                      'test_cmd': {'a': 'uiae', 'b': 123, 'c': {'d': 1337}},
                      **reference_default_config}


def test_recursive_update0(basic_config_dict):
    current = basic_config_dict
    new = {'a': 5}
    # run
    OptionsHandler._recursive_update(current, new)
    # check
    expected = basic_config_dict
    expected['a'] = 5
    assert repr(current) == repr(expected)


def test_recursive_update1(basic_config_dict):
    current = basic_config_dict
    new = {'a': 5, 'h': 71}
    # run
    OptionsHandler._recursive_update(current, new)
    # check
    expected = basic_config_dict
    expected['a'] = 5
    expected['h'] = 71
    assert repr(current) == repr(expected)


def test_recursive_update2(basic_config_dict):
    current = basic_config_dict
    new = {'a': {'c': {'c': {'f': 'blub'}}}}
    # run
    OptionsHandler._recursive_update(current, new)
    # check
    expected = basic_config_dict
    expected['a']['c']['c']['f'] = 'blub'
    assert repr(current) == repr(expected)


def test_recursive_update3(basic_config_dict):
    current = basic_config_dict
    new = {'a': {'c': None}}
    # run
    OptionsHandler._recursive_update(current, new)
    # check
    expected = basic_config_dict
    expected['a']['c'] = None
    assert repr(current) == repr(expected)


def test_recursive_update4(basic_config_dict):
    current = basic_config_dict
    new = {'a': {'c': {'d': {42: 'test'}}}}
    # run
    OptionsHandler._recursive_update(current, new)
    # check
    expected = basic_config_dict
    expected['a']['c']['d'] = {42: 'test'}
    assert repr(current) == repr(expected)


def test_recursive_update5(basic_config_dict):
    current = basic_config_dict
    current['a'] = {'c': None}
    new = {'a': {'c': {'d': {42: 'test'}}}}
    # run
    OptionsHandler._recursive_update(current, new)
    expected = basic_config_dict
    expected['a']['c']['d'] = {42: 'test'}
    assert current == expected


def test_recursive_update6(basic_config_dict):
    current = basic_config_dict
    new = {'x': {'c': {'d': {42: 'test'}}}}
    # run
    with pytest.raises(KeyError):
        OptionsHandler._recursive_update(current, new)


def test_handle_cmd_specification_none(opt_handler_two_commands):
    oh = opt_handler_two_commands
    cmd_args = {'mode': 'batch'}
    result = oh._handle_cmd_specification(cmd_args)
    assert result == {'mode': 'batch'}


def test_handle_cmd_specification_single(opt_handler_two_commands):
    oh = opt_handler_two_commands
    cmd_args = {'other_cmd': {'one': 'blub'}}
    result = oh._handle_cmd_specification(cmd_args)
    assert result == {'other_cmd': {'one': 'blub'}, 'test_cmd': None}


def test_handle_cmd_specification_double(opt_handler_two_commands):
    oh = opt_handler_two_commands
    cmd_args = {'other_cmd': {'one': 'blub'},
                'test_cmd': {'b': 69}}
    result = oh._handle_cmd_specification(cmd_args)
    assert result == {'other_cmd': {'one': 'blub'}, 'test_cmd': {'b': 69}}
