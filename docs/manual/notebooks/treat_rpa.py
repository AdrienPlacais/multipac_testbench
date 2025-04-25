#!/usr/bin/env python3
"""Define a classic workflow to study the RPA signals."""
import tomllib
from pathlib import Path
from pprint import pformat, pprint

import matplotlib.pyplot as plt
import multipac_testbench.instruments as ins
from multipac_testbench.multipactor_test import MultipactorTest

if __name__ == "__main__":
    project = Path("../data/campaign_ERPA")
    config_path = Path(project, "testbench_configuration.toml")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    results_path = Path(project, "MVE5-120MHz-50Ohm-BDTcomp-ERPA1_0dBm.csv")
    multipactor_test = MultipactorTest(
        results_path, config, freq_mhz=120.0, swr=1.0, sep=","
    )

    # Plot RPA current vs RPA potential
    _, _ = multipactor_test.sweet_plot(
        ins.RPACurrent, xdata=ins.RPAPotential, tail=1200
    )
