import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest

from mlfaker.generators import BaseGenerator, CategoricalGenerator, NormalGenerator


@pytest.fixture
def rand_sr():
    """Random normally distributed series"""
    return pd.Series(np.random.RandomState(seed=1).normal(0, 1, 10), name="test")


@pytest.fixture
def base_gen():
    """Instatiated BaseGenerator"""
    return BaseGenerator(data_name="foo", fillrate=0.5, seed=1)


def test_base_generator():
    """Base class constructor test"""
    name = "databoi"
    fillrate = 1.0
    b = BaseGenerator(data_name=name, fillrate=fillrate, seed=1)
    assert b.fillrate == fillrate
    assert b.data_name == name


def test_bad_fillrate_init():
    """Test fill rate between 0 and 1 in init"""
    with pytest.raises(ValueError):
        BaseGenerator(data_name="foo", fillrate=10, seed=1)

    with pytest.raises(ValueError):
        BaseGenerator(data_name="foo", fillrate=-1, seed=1)


def test_bad_fillrate_set(base_gen):
    """Test fill rate between 0 and 1 for setter"""
    with pytest.raises(ValueError):
        base_gen.fillrate = -1

    with pytest.raises(ValueError):
        base_gen.fillrate = 2


def test_nuller(base_gen, rand_sr):
    """Test nuller is working"""
    for frate in [0.2, 0.5, 0.6]:
        base_gen.fillrate = frate
        fnull = base_gen._nuller(rand_sr).isna().mean()
        np.testing.assert_almost_equal(1 - fnull, frate)


@pytest.mark.parametrize(
    "gen, size, tcheck",
    [
        (NormalGenerator("foo", 0.5), 18, ptypes.is_numeric_dtype),
        (CategoricalGenerator("foo", 0.5, classes=[1, 2]), 11, ptypes.is_numeric_dtype),
        (
            CategoricalGenerator("foo", 0.5, classes=["bam", "booz", "led"]),
            13,
            ptypes.is_string_dtype,
        ),
    ],
)
def test_numerical_generator(gen, size, tcheck):
    """Test size and type of generated data"""
    out = gen.generate(size)
    assert tcheck(out)
    assert len(out) == size


def test_raise_classes_rate():
    with pytest.raises(ValueError):
        CategoricalGenerator(data_name="foo", classes=[0, 1, 2], rates=[0.2, 0.3])
