import pytest
from calculon.utils.validoptions import FailedCastError, FailedVerificationError, ValidOption

def test_option_bool():
    vo = ValidOption(type=bool, default=True)
    assert vo.cast(0) is False
    assert vo.cast(1) is True
    assert vo.cast('true') is True
    assert vo.cast('false') is False
    with pytest.raises(FailedCastError):
        vo.cast('l')
    vo.verify(False)
    vo.verify(5)


def test_option_int():
    vo = ValidOption(type=int, default=5, choices=[1, 2, 3, 5, 6])
    assert vo.cast('6') == 6
    assert vo.cast('4') == 4
    with pytest.raises(FailedCastError):
        vo.cast('l')
    vo.verify(6)
    with pytest.raises(FailedVerificationError):
        vo.verify(4)


def test_option_float():
    vo = ValidOption(type=float, default=3.14, choices=[1.2, 3.14])
    assert vo.cast('-0.2') == -0.2
    with pytest.raises(FailedCastError):
        vo.cast('l')
    vo.verify(1.2)
    with pytest.raises(FailedVerificationError):
        vo.verify(1.3)


def test_option_str():
    with pytest.raises(FailedVerificationError):
        ValidOption(type=str, default='hi', choices=['blah', 'blub'])
    vo = ValidOption(type=str, default='blah', choices=['blah', 'blub'])
    assert vo.cast('test') == 'test'
    vo.verify('blub')
    with pytest.raises(FailedVerificationError):
        vo.verify('foo')


def test_option_list_int():
    vo = ValidOption(type=list, subtype=int, default=[1, 2],
                     choices=[1, 2, 3, 4])
    assert vo.cast(['3', '4']) == [3, 4]
    assert vo.cast('[3,4]') == [3, 4]
    assert vo.cast('[3, 4]') == [3, 4]
    vo.verify([3, 4])
    with pytest.raises(FailedVerificationError):
        vo.verify(3)
    with pytest.raises(FailedVerificationError):
        vo.verify(['3', 4])
    with pytest.raises(FailedVerificationError):
        vo.verify([3, 4, 5])


def test_option_list_str_len():
    vo = ValidOption(type=list, subtype=str, default=['foo', 'bar'], length=2,
                     choices=['foo', 'bar', 'blah', 'blub'])
    assert vo.cast(['bar', 'blub']) == ['bar', 'blub']
    assert vo.cast('[bar,blub]') == ['bar', 'blub']
    assert vo.cast('[bar, blub]') == ['bar', 'blub']
    vo.verify(['bar', 'blub'])
    with pytest.raises(FailedVerificationError):
        vo.verify(['bar'])
    with pytest.raises(FailedVerificationError):
        vo.verify(['bar', 'blah', 'blub'])
    with pytest.raises(FailedVerificationError):
        vo.verify(['bar', 'broken'])