from abc import abstractmethod
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


class BaseGenerator:
    """Base class for the generators

    Includes some checks for fill rates and a _nuller static method

    Args:
        name: name for the generated data
        fillrate: fillrate (1-fraction NaN)
    """

    def __init__(self, name: str, fillrate: float):
        self.name = name
        self.fillrate = fillrate

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
            sr.loc[sr.sample(frac=1 - self.fillrate).index] = np.NaN
        return sr

    @abstractmethod
    def generate(self, length: int) -> pd.Series:
        """Data generation method"""
        raise NotImplementedError()


class NumGenerator(BaseGenerator):
    """Numerical (normally distributed) data generator

    Args:
        name: name for the generated data (series)
        fillrate: fillrate (1-fraction NaN)
        mu: mean value
        sig: standard deviation
    """

    def __init__(
        self, name: str, fillrate: float = 1.0, mu: float = 0.0, sig: float = 1.0
    ):
        super().__init__(name, fillrate)
        self.mu = mu
        self.sig = sig

    def generate(self, length: int) -> pd.Series:
        """Generates numerical data series"""
        self.value = self._nuller(
            pd.Series(np.random.normal(self.mu, self.sig, length), name=self.name)
        )
        return self._nuller(
            pd.Series(np.random.normal(self.mu, self.sig, length), name=self.name)
        )


class CatGenerator(BaseGenerator):
    """Categorical data generator

    Args:
        name: name for the generated data (series)
        fillrate: fillrate (1-fraction NaN)
        classes: categorical class, e.g. ["foo", "bar"] or [0, 1, 3]
        rates: rates of the classes, e.g., [0.1, 0.9]
    """

    def __init__(
        self,
        name: str,
        fillrate: float = 1,
        classes: Union[Sequence[int], Sequence[str]] = [0, 1],
        rates: Optional[Sequence[float]] = None,
    ):
        super().__init__(name, fillrate)
        if rates is not None and len(classes) != len(rates):
            raise ValueError(
                "The number of classes much match the rate array of probabilities"
            )
        else:
            self.rates = rates
        self.classes = classes

    def generate(self, length) -> pd.Series:
        """Generate categorical data series"""
        return self._nuller(
            pd.Series(
                np.random.choice(self.classes, length, p=self.rates), name=self.name
            )
        )
