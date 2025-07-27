"""This package stores all the function to study MULTIPAC testbench."""

import importlib.metadata

__version__ = importlib.metadata.version("multipac_testbench")

from multipac_testbench.multipactor_test.multipactor_test import (
    MultipactorTest,
)
from multipac_testbench.multipactor_test.power_step import (
    PowerStep,
    PowerStepSet,
)
from multipac_testbench.test_campaign import TestCampaign

__all__ = ["MultipactorTest", "TestCampaign", "PowerStep", "PowerStepSet"]
