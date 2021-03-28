from typing import Optional, Sequence

import numpy as np
import pandas as pd


class BaseGenerator:
    """TODO"""

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

    def _nuller(self, sr):
        sr = sr.copy()
        if self.fillrate != 1:
            sr.loc[sr.sample(frac=1 - self.fillrate).index] = np.NaN
        return sr


class NumGenerator(BaseGenerator):
    """TODO"""

    def __init__(
        self, name: str, fillrate: float = 1.0, mu: float = 0.0, sig: float = 1.0
    ):
        super().__init__(name, fillrate)
        self.mu = mu
        self.sig = sig

    def generate(self, length):
        self.value = self._nuller(
            pd.Series(np.random.normal(self.mu, self.sig, length), name=self.name)
        )
        return self._nuller(
            pd.Series(np.random.normal(self.mu, self.sig, length), name=self.name)
        )


class CatGenerator(BaseGenerator):
    """TODO"""

    def __init__(
        self,
        name: str,
        fillrate: float = 1,
        classes: Sequence = [0, 1],
        rates: Optional[Sequence] = None,
    ):
        super().__init__(name, fillrate)
        if rates is not None and len(classes) != len(rates):
            raise ValueError(
                "The number of classes much match the rate array of probabilities"
            )
        else:
            self.rates = rates
        self.classes = classes

    def generate(self, length):
        return self._nuller(
            pd.Series(
                np.random.choice(self.classes, length, p=self.rates), name=self.name
            )
        )
