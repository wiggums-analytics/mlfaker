from functools import partial
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


class BaseGenerator:
    """Base class for the generators

    Includes some checks for fill rates and a _nuller static method

    Args:
        data_name: name for the generated data
        fillrate: fillrate (1-fraction NaN)
    """

    def __init__(self, data_name: str, fillrate: float = 1.0, seed: int = 1):
        self.data_name = data_name
        self.fillrate = fillrate
        self.seed = seed
        self.rs = np.random.RandomState(seed)
        self.generator = None

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

    def set_generator(self, generator_name, **kwargs):
        self.generator_name = generator_name
        self.generator = partial(getattr(self.rs, generator_name), **kwargs)

    def generate(self, size: int) -> pd.Series:
        """Data generation method"""
        if self.generator is None:
            raise ValueError("You must set generator with set_generator method")
        return self._nuller(pd.Series(self.generator(size=size), name=self.data_name))


class NormalGenerator(BaseGenerator):
    """Normally data generator

    Args:
        data_name: name for the generated data (series)
        fillrate: fillrate (1-fraction NaN)
        loc: mean value
        scale: standard deviation
    """

    generator_name = "normal"

    def __init__(
        self,
        data_name: str,
        fillrate: float = 1.0,
        loc: float = 0.0,
        scale: float = 1.0,
        seed: int = 1,
    ):
        super().__init__(
            data_name=data_name,
            fillrate=fillrate,
            seed=seed,
        )
        self.loc = loc
        self.scale = scale
        self.set_generator(self.generator_name, loc=self.loc, scale=self.scale)


class CategoricalGenerator(BaseGenerator):
    """Categorical data generator

    Args:
        data_name: name for the generated data (series)
        fillrate: fillrate (1-fraction NaN)
        classes: categorical class, e.g. ["foo", "bar"] or [0, 1, 3]
        rates: rates of the classes, e.g., [0.1, 0.9]
    """

    generator_name = "choice"

    def __init__(
        self,
        data_name: str,
        fillrate: float = 1,
        classes: Union[Sequence[int], Sequence[str]] = [0, 1],
        rates: Optional[Sequence[float]] = None,
        seed: int = 1,
    ):
        if rates is None or len(classes) == len(rates):
            self.rates = rates
        else:
            raise ValueError(
                "The number of classes much match the rate array of probabilities"
            )
        self.classes = classes
        super().__init__(
            data_name=data_name,
            fillrate=fillrate,
            seed=seed,
        )
        self.set_generator(self.generator_name, a=self.classes, p=self.rates)
