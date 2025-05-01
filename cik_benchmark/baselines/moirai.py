import pandas as pd
import numpy as np
import time
from .base import Baseline
from ..base import BaseTask

import torch
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from einops import rearrange

from transformers import set_seed


class MoiraiForecaster(Baseline):

    __version__ = "0.1.0"  # Modification will trigger re-caching

    def __init__(
        self,
        model_size,
        patch_size=16,
        batch_size=32,
        num_parallel_samples=100,
        bagging_size=10,
        seed=42,
    ):
        """
        Get predictions from a Moirai model.

        Notes:
        ------
        This model requires a seasonal periodicity, which it currently gets from a
        hard coded association from the data index frequency (hourly -> 24 hours periods).
        """
        self.model_size = model_size
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_parallel_samples = num_parallel_samples
        self.bagging_size = bagging_size
        self.seed = seed
        super().__init__()

    def __call__(self, task_instance, n_samples: int) -> np.ndarray:
        
        raise NotImplementedError

    def forecast(
        self,
        task_instance,
        n_samples: int,
    ) -> np.ndarray:
        """
        This method allows a forecast to be done without requiring a complete BaseTask instance.
        This is primarily meant to be called inside a BaseTask constructor when doing rejection sampling or similar approaches.
        """
        # If there is no period, then disable the seasonal component of the model (seasonal_periods will be ignored)

        set_seed(self.seed) # DO NOT REMOVE THIS LINE

        raise NotImplementedError

    @property
    def cache_name(self) -> str:
        return f"{self.__class__.__name__}_{self.model_size}"


# if __name__ == "__main__":
#     # Dummy example to run the model
#     class DummyTask:
#         def __init__(self):
#             self.past_time = pd.Series(
#                 np.random.randn(100), index=pd.date_range("20210101", periods=100)
#             )
#             self.future_time = pd.Series(
#                 np.random.randn(10), index=pd.date_range("20210501", periods=10)
#             )

#     task_instance = DummyTask()
#     forecaster = MoiraiForecaster(
#         model_size="small",
#         context_length=100,
#         prediction_length=len(task_instance.future_time),
#         patch_size=16,
#         batch_size=32,
#         num_parallel_samples=100,
#         bagging_size=10,
#     )
#     predictions = forecaster(task_instance, n_samples=50)
#     print(predictions)
