import pytest

from mlfaker.generators import BaseGenerator


@pytest.fixture
def base_gen():
    return BaseGenerator(name="foo", fillrate=0.5)


def test_base_generator():
    """Base class constructor test"""
    name = "databoi"
    fillrate = 1.0
    b = BaseGenerator(name=name, fillrate=fillrate)
    assert b.fillrate == fillrate
    assert b.name == name


def test_bad_fillrate_init():
    """Test fill rate between 0 and 1 in init"""
    with pytest.raises(ValueError):
        b = BaseGenerator(name="foo", fillrate=10)

    with pytest.raises(ValueError):
        b = BaseGenerator(name="foo", fillrate=-1)


def test_bad_fillrate_set(base_gen):
    with pytest.raises(ValueError):
        base_gen.fillrate = -1

    with pytest.raises(ValueError):
        base_gen.fillrate = 2
