from abc import abstractmethod
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

    def __init__(self, data_name: str, fillrate: float, seed=1, **gen_kwargs):
        # def __init__(
        # self, generator_name, data_name: str, fillrate: float, seed=1, **gen_kwargs
        # ):
        self.data_name = data_name
        self.fillrate = fillrate
        self.seed = seed
        self.rs = np.random.RandomState(seed)
        # self.gen_kwargs = gen_kwargs
        # self.generator_name = generator_name

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

    # def generate(self, size: int) -> pd.Series:
    # """Data generation method"""
    # generator = partial(getattr(self.rs, self.generator_name), **self.gen_kwargs)
    # return self._nuller(pd.Series(generator(size=size), name=self.data_name))

    def _generate(self, generator_name, size: int, **kwargs) -> pd.Series:
        """Data generation method"""
        generator = partial(getattr(self.rs, generator_name), **kwargs)
        return self._nuller(pd.Series(generator(size=size), name=self.data_name))

    @abstractmethod
    def generate(self, size: int) -> pd.Series:
        """Data generation method"""
        raise NotImplementedError()


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

    def generate(self, size: int) -> pd.Series:
        """Data generation method"""
        return super()._generate(
            self.generator_name, size=size, loc=self.loc, scale=self.scale
        )


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
        if rates is not None and len(classes) != len(rates):
            raise ValueError(
                "The number of classes much match the rate array of probabilities"
            )
        else:
            self.rates = rates
        self.classes = classes
        super().__init__(
            data_name=data_name,
            fillrate=fillrate,
            seed=seed,
        )

    def generate(self, size: int) -> pd.Series:
        """Data generation method"""
        return super()._generate(
            self.generator_name, size=size, a=self.classes, p=self.rates
        )
