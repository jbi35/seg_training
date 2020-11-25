import pytest
import argparse

from calculon.utils.parserhandler import ParserHandler
from calculon.utils.validoptions import ValidOption

@pytest.fixture(scope='module')
def sample_flat_options():
    basic = {'a': ValidOption(type=int, default=4),
             'outer': {'c': ValidOption(type=int, default=33),
                       'inner:blub': ValidOption(type=str, default='asdf',
                                                 choices=['asdf', 'uiae',
                                                          'cast string']),
                       'inner:num': ValidOption(type=float, default=3.14159),
                       'inner:list': ValidOption(type=list, subtype=int,
                                                 length=2, default=[1, 2])}}
    return basic


def test_parse_multi_args_inner(sample_flat_options):
    ph = ParserHandler(sample_flat_options)
    result = ph._parse_multi_args(['PROG.py', '--a', '23'])
    assert result == argparse.Namespace(a=23, outer=None)


def test_parse_multi_args_outer(sample_flat_options):
    ph = ParserHandler(sample_flat_options)
    result = ph._parse_multi_args(['PROG.py', 'outer', '--c', '23'])
    assert result == argparse.Namespace(a=None, outer=argparse.Namespace(
        a=None, c=23, **{'inner:blub': None, 'inner:num': None,
                         'inner:list': None}))


def test_unpack_namespace():
    namespace = argparse.Namespace(a=None, outer=argparse.Namespace(
            a=None, c=23, **{'inner:blub': None, 'inner:num': 3.14}))
    result = ParserHandler._unpack_namespace(namespace)
    assert result == {'outer': {'c': 23, 'inner:num': 3.14}}


def test_parse_args(sample_flat_options):
    ph = ParserHandler(sample_flat_options)
    result = ph.parse_args(['PROG.py', 'outer', '--c', '23', '--inner:num',
                            '3.14', '--inner:list', '5', '6'])
    print(result)
    assert result == {'outer': {'c': 23, 'inner:num': 3.14,
                                'inner:list': [5, 6]}}


def test_parse_args_wrong_length(sample_flat_options):
    ph = ParserHandler(sample_flat_options)
    with pytest.raises(SystemExit):
        ph.parse_args(['PROG.py', 'outer', '--c', '23', '--inner:list',
                       '5'])
    with pytest.raises(SystemExit):
        ph.parse_args(['PROG.py', 'outer', '--c', '23', '--inner:list',
                       '5', '6', '7'])
