import pytest
import calculon.utils.log as log

def test_split_colors_plain():
    msg = ('blub',)
    plain, color = log._split_color_info(msg)
    assert plain == 'blub'
    assert color == 'blub'


def test_split_colors_plain_multi():
    msg = ('blub', 'blah', 'grrr')
    plain, color = log._split_color_info(msg)
    assert plain == 'blub blah grrr'
    assert color == 'blub blah grrr'


def test_split_colors_blue_multi():
    msg = ('blub', 'BLUE', 'grrr')
    plain, color = log._split_color_info(msg)
    assert plain == 'blub grrr'
    assert color == 'blub\033[94m grrr\033[0m'


def test_split_colors_blue_green_multi():
    msg = ('blub', 'BLUE', 'grrr', 'GREEN', 'paff')
    plain, color = log._split_color_info(msg)
    assert plain == 'blub grrr paff'
    assert color == 'blub\033[94m grrr\033[92m paff\033[0m'


def test_split_colors_misspelled_color():
    msg = ('blub', 'BLUE', 'grrr', 'GRENN', 'paff')
    plain, color = log._split_color_info(msg)
    assert plain == 'blub grrr GRENN paff'
    assert color == 'blub\033[94m grrr GRENN paff\033[0m'


def test_split_colors_lower_case_color():
    msg = ('blub', 'blue', 'grrr', 'GRENN', 'paff')
    plain, color = log._split_color_info(msg)
    assert plain == 'blub blue grrr GRENN paff'
    assert color == 'blub blue grrr GRENN paff'


def test_split_colors_color_only():
    msg = ('BLUE',)
    plain, color = log._split_color_info(msg)
    assert plain == 'BLUE'
    assert color == 'BLUE'


def test_split_colors_empty_string():
    msg = ('',)
    plain, color = log._split_color_info(msg)
    assert plain == ''
    assert color == ''


def test_split_colors_empty_tuple():
    msg = ()
    plain, color = log._split_color_info(msg)
    assert plain == ''
    assert color == ''


def test_split_colors_wrong_type():
    msg = 5
    with pytest.raises(TypeError):
        log._split_color_info(msg)
