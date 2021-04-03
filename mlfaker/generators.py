from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd


class BaseGenerator(metaclass=ABCMeta):
    """Base class for the generators

    Includes some checks for fill rates and a _nuller static method

    Args:
        name: name for the generated data
        fillrate: fillrate (1-fraction NaN)
    """

    generator_name = None

    def __init__(self, name: str, fillrate: float, seed=1, **gen_kwargs):
        self.name = name
        self.fillrate = fillrate
        self.seed = seed
        self.rs = np.random.RandomState(seed)
        self.generator = partial(getattr(self.rs, self.generator_name), **gen_kwargs)

    @property
    def fillrate(self):
        return self._fillrate

    @fillrate.setter
    def fillrate(self, value: float):
        if value < 0 or value > 1:
            raise ValueError("Fillrate must be between 0 and 1")
        self._fillrate = value

    def _nuller(self, sr: pd.Series) -> pd.Series:
        """Returns a nulled copy of a series"""
        sr = sr.copy()
        if self.fillrate != 1:
            sr.loc[
                sr.sample(frac=1 - self.fillrate, random_state=self.rs).index
            ] = np.NaN
        return sr

    def generate(self, size: int) -> pd.Series:
        """Data generation method"""
        return self._nuller(pd.Series(self.generator(size=size), name=self.name))


class NormalGenerator(BaseGenerator):
    """Numerical (normally distributed) data generator

    Args:
        name: name for the generated data (series)
        fillrate: fillrate (1-fraction NaN)
        loc: mean value
        scale: standard deviation
    """

    generator_name = "normal"

    def __init__(
        self, name: str, fillrate: float = 1.0, loc: float = 0.0, scale: float = 1.0
    ):
        super().__init__(name, fillrate, loc=loc, scale=scale)


class NormalGenerator(BaseGenerator):
    """Numerical (normally distributed) data generator

    Args:
        name: name for the generated data (series)
        fillrate: fillrate (1-fraction NaN)
        loc: mean value
        scale: standard deviation
    """

    generator_name = "normal"

    def __init__(
        self, name: str, fillrate: float = 1.0, loc: float = 0.0, scale: float = 1.0
    ):
        super().__init__(name, fillrate, loc=loc, scale=scale)


class CategorialGenerator(BaseGenerator):
    """Categorical data generator

    Args:
        name: name for the generated data (series)
        fillrate: fillrate (1-fraction NaN)
        classes: categorical class, e.g. ["foo", "bar"] or [0, 1, 3]
        rates: rates of the classes, e.g., [0.1, 0.9]
    """

    generator_name = "choice"

    def __init__(
        self,
        name: str,
        fillrate: float = 1,
        classes: Union[Sequence[int], Sequence[str]] = [0, 1],
        rates: Optional[Sequence[float]] = None,
    ):
        if rates is not None and len(classes) != len(rates):
            raise ValueError(
                "The number of classes much match the rate array of probabilities"
            )
        else:
            self.rates = rates
        self.classes = classes
        super().__init__(name, fillrate, a=classes, p=rates)
